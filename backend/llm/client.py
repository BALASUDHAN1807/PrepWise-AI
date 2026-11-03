"""ONE provider-agnostic LLM wrapper used by every module.

Supported providers (selected with ``LLM_PROVIDER`` in ``.env``):

* ``gemini``    - Google Gemini REST API (``generateContent``)
* ``openai``    - OpenAI Chat Completions API (or any compatible base URL)
* ``anthropic`` - Anthropic Messages API
* ``ollama``    - local Llama models served by Ollama
* ``mock``      - deterministic rule-based stand-ins for CI and offline demos

Design rules enforced here:

* Every call that feeds application logic goes through :meth:`LLMClient.complete_json`,
  which validates the JSON against a Pydantic model, retries up to
  ``LLM_MAX_RETRIES`` times (default 2) with the validation error fed back to the
  model, and finally returns a *safe fallback* instead of crashing.
* There is no tool calling and no autonomous loop: one request in, one response out.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from backend.core.config import Settings, get_settings

log = logging.getLogger("prepwise.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a provider call fails and no fallback is available."""


@dataclass
class LLMResult:
    """Raw text completion plus metadata."""

    text: str
    provider: str
    model: str
    latency_ms: float


@dataclass
class LLMJSONResult(Generic[T]):
    """Validated structured output plus metadata about how it was obtained."""

    data: T
    provider: str
    model: str
    latency_ms: float
    attempts: int
    used_fallback: bool = False
    error: str | None = None
    raw_text: str = ""
    meta: dict = field(default_factory=dict)

    def info(self) -> dict:
        """Small JSON-serialisable summary stored next to results for transparency."""
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "attempts": self.attempts,
            "used_fallback": self.used_fallback,
            "error": self.error,
        }


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and leading prose."""
    if text is None:
        raise ValueError("empty response")
    candidate = text.strip()
    m = _FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] block.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = candidate.find(open_c), candidate.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("response is not valid JSON")


def schema_hint(schema: type[BaseModel]) -> str:
    """Compact JSON schema string appended to prompts so models know the shape."""
    return json.dumps(schema.model_json_schema(), separators=(",", ":"))


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
class BaseProvider:
    name = "base"

    def __init__(self, settings: Settings, model: str | None = None):
        self.settings = settings
        self.model = model or self.default_model()

    def default_model(self) -> str:  # pragma: no cover - overridden
        return ""

    def generate(
        self,
        system: str,
        user: str,
        temperature: float,
        json_mode: bool,
        task: str | None = None,
        variables: dict | None = None,
    ) -> str:
        raise NotImplementedError

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            with httpx.Client(timeout=self.settings.llm_timeout_sec) as client:
                resp = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: network error: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()


class GeminiProvider(BaseProvider):
    name = "gemini"

    def default_model(self) -> str:
        return self.settings.llm_model or self.settings.gemini_model

    def generate(self, system, user, temperature, json_mode, task=None, variables=None) -> str:
        if not self.settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        gen_cfg: dict[str, Any] = {"temperature": temperature}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen_cfg,
        }
        data = self._post(url, {"x-goog-api-key": self.settings.gemini_api_key}, payload)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError) as exc:
            raise LLMError(f"gemini: unexpected response shape: {str(data)[:200]}") from exc


class OpenAIProvider(BaseProvider):
    name = "openai"

    def default_model(self) -> str:
        return self.settings.llm_model or self.settings.openai_model

    def generate(self, system, user, temperature, json_mode, task=None, variables=None) -> str:
        if not self.settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post(url, headers, payload)
        except LLMError as exc:
            # Some reasoning models only accept the default temperature.
            if "temperature" not in str(exc):
                raise
            payload.pop("temperature", None)
            data = self._post(url, headers, payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"openai: unexpected response shape: {str(data)[:200]}") from exc


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def default_model(self) -> str:
        return self.settings.llm_model or self.settings.anthropic_model

    def generate(self, system, user, temperature, json_mode, task=None, variables=None) -> str:
        if not self.settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        headers = {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        }
        if json_mode:
            system = system + "\n\nRespond with a single valid JSON object and nothing else."
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        data = self._post("https://api.anthropic.com/v1/messages", headers, payload)
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError(f"anthropic: unexpected response shape: {str(data)[:200]}") from exc


class OllamaProvider(BaseProvider):
    name = "ollama"

    def default_model(self) -> str:
        return self.settings.llm_model or self.settings.ollama_model

    def generate(self, system, user, temperature, json_mode, task=None, variables=None) -> str:
        url = self.settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        data = self._post(url, {}, payload)
        try:
            return data["message"]["content"]
        except KeyError as exc:
            raise LLMError(f"ollama: unexpected response shape: {str(data)[:200]}") from exc


class MockProvider(BaseProvider):
    """Deterministic rule-based provider used in CI and when no API key is set.

    It returns schema-valid JSON produced by ``backend.llm.mock_responses`` (which
    reuses the same heuristics the app uses as fallbacks). It is NOT a language
    model; quality-sensitive results must be produced with a real provider.
    """

    name = "mock"

    def default_model(self) -> str:
        return "rule-based-v1"

    def generate(self, system, user, temperature, json_mode, task=None, variables=None) -> str:
        from backend.llm.mock_responses import respond

        return json.dumps(respond(task or "", variables or {}, system, user))


PROVIDERS: dict[str, type[BaseProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
}


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class LLMClient:
    """Facade over the configured provider with JSON validation and retries."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        name = (provider or self.settings.llm_provider).lower().strip()
        if name not in PROVIDERS:
            raise LLMError(f"Unknown LLM provider '{name}'. Choose one of {sorted(PROVIDERS)}")
        self.provider = PROVIDERS[name](self.settings, model)

    @property
    def provider_name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        json_mode: bool = False,
        task: str | None = None,
        variables: dict | None = None,
    ) -> LLMResult:
        """Single text completion (no validation). Raises :class:`LLMError`."""
        temp = self.settings.temperature_eval if temperature is None else temperature
        start = time.perf_counter()
        text = self.provider.generate(system, user, temp, json_mode, task, variables)
        latency = (time.perf_counter() - start) * 1000
        log.info("llm call provider=%s model=%s task=%s %.0fms", self.provider_name, self.model, task, latency)
        return LLMResult(text=text, provider=self.provider_name, model=self.model, latency_ms=latency)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
        task: str | None = None,
        variables: dict | None = None,
        fallback: T | Callable[[], T] | None = None,
        max_retries: int | None = None,
    ) -> LLMJSONResult[T]:
        """Return output validated against ``schema``.

        Retries up to ``max_retries`` (default ``LLM_MAX_RETRIES`` = 2) times on
        provider errors, invalid JSON or schema violations; each retry tells the
        model what was wrong. If every attempt fails, ``fallback`` is returned
        (flagged with ``used_fallback=True``); without a fallback, LLMError is raised.
        """
        retries = self.settings.llm_max_retries if max_retries is None else max_retries
        system_full = (
            system.rstrip()
            + "\n\nReturn ONLY a JSON object that matches this JSON schema "
            "(no markdown, no commentary):\n"
            + schema_hint(schema)
        )
        user_msg = user
        last_error: str | None = None
        last_text = ""
        total_latency = 0.0
        attempts = 0
        for _ in range(1 + retries):
            attempts += 1
            try:
                res = self.complete(system_full, user_msg, temperature, True, task, variables)
            except LLMError as exc:
                last_error = str(exc)
                log.warning("llm provider error (attempt %d): %s", attempts, last_error)
                if "is not set" in last_error:  # no point retrying a missing key
                    break
                continue
            total_latency += res.latency_ms
            last_text = res.text
            try:
                data = schema.model_validate(extract_json(res.text))
                return LLMJSONResult(
                    data=data,
                    provider=self.provider_name,
                    model=self.model,
                    latency_ms=total_latency,
                    attempts=attempts,
                    raw_text=res.text,
                )
            except (ValueError, ValidationError) as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                log.warning("invalid JSON from llm (attempt %d): %s", attempts, last_error[:200])
                user_msg = (
                    user
                    + "\n\nYour previous reply was rejected because it was not valid JSON for "
                    f"the required schema. Error: {last_error}\n"
                    "Reply again with ONLY the corrected JSON object."
                )
        if fallback is not None:
            data = fallback() if callable(fallback) else fallback
            return LLMJSONResult(
                data=data,
                provider=self.provider_name,
                model=self.model,
                latency_ms=total_latency,
                attempts=attempts,
                used_fallback=True,
                error=last_error,
                raw_text=last_text,
            )
        raise LLMError(f"LLM output invalid after {attempts} attempts: {last_error}")


def get_llm(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Convenience constructor used by services."""
    return LLMClient(provider=provider, model=model)
