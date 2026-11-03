"""Load versioned prompt templates from ``backend/llm/prompts``.

Each template lives in its own file named ``<name>.v<N>.yaml`` with the keys
``name, version, description, temperature, output_schema, system, user``.
Templates are rendered with Jinja2 (``{{ variable }}``) using StrictUndefined so
a missing variable fails loudly instead of silently producing a bad prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from backend.core.config import get_settings

_FILE_RE = re.compile(r"^(?P<name>[a-z0-9_]+)\.v(?P<version>\d+)\.yaml$")
_env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)


@dataclass
class PromptTemplate:
    name: str
    version: int
    description: str
    temperature: float
    output_schema: str
    system: str
    user: str
    path: str = ""

    def render(self, **variables) -> tuple[str, str]:
        """Return ``(system, user)`` with variables substituted."""
        return (
            _env.from_string(self.system).render(**variables),
            _env.from_string(self.user).render(**variables),
        )

    @property
    def id(self) -> str:
        return f"{self.name}.v{self.version}"


def _prompts_dir() -> Path:
    return Path(get_settings().prompts_dir)


@lru_cache(maxsize=1)
def _index() -> dict[str, dict[int, Path]]:
    idx: dict[str, dict[int, Path]] = {}
    for p in sorted(_prompts_dir().glob("*.yaml")):
        m = _FILE_RE.match(p.name)
        if m:
            idx.setdefault(m["name"], {})[int(m["version"])] = p
    return idx


def list_prompts() -> dict[str, list[int]]:
    """Map of prompt name -> available versions (ascending)."""
    return {name: sorted(versions) for name, versions in _index().items()}


def load_prompt(name: str, version: int | None = None) -> PromptTemplate:
    """Load a prompt; the latest version is used when ``version`` is None."""
    versions = _index().get(name)
    if not versions:
        raise KeyError(f"Unknown prompt '{name}'")
    v = max(versions) if version is None else version
    if v not in versions:
        raise KeyError(f"Prompt '{name}' has no version {v}")
    raw = yaml.safe_load(versions[v].read_text(encoding="utf-8"))
    return PromptTemplate(
        name=raw["name"],
        version=int(raw["version"]),
        description=raw.get("description", ""),
        temperature=float(raw.get("temperature", get_settings().temperature_eval)),
        output_schema=raw.get("output_schema", "GenericJSON"),
        system=raw["system"],
        user=raw["user"],
        path=str(versions[v]),
    )


def reload_prompts() -> None:
    """Clear the file index cache (used after adding prompt files)."""
    _index.cache_clear()
