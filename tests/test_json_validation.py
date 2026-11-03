"""LLM wrapper: JSON extraction, Pydantic validation, retries (max 2) and safe fallbacks."""
import json

import pytest
from pydantic import ValidationError

from backend.core.config import get_settings
from backend.llm.client import PROVIDERS, BaseProvider, LLMClient, LLMError, extract_json
from backend.llm.prompt_loader import list_prompts, load_prompt
from backend.models import llm_schemas as S


class ScriptedProvider(BaseProvider):
    """Returns queued replies in order and counts calls."""
    name = "scripted"
    replies: list = []
    calls = 0

    def default_model(self):
        return "scripted-1"

    def generate(self, system, user, temperature, json_mode, task=None, variables=None):
        ScriptedProvider.calls += 1
        r = ScriptedProvider.replies[min(ScriptedProvider.calls - 1, len(ScriptedProvider.replies) - 1)]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture()
def scripted(monkeypatch):
    monkeypatch.setitem(PROVIDERS, "scripted", ScriptedProvider)
    ScriptedProvider.calls = 0

    def make(*replies):
        ScriptedProvider.replies = list(replies)
        return LLMClient(provider="scripted", settings=get_settings())
    return make


VALID = json.dumps({"summary": "ok", "suggestions": ["Do X"]})
FALLBACK = S.ResumeSuggestions(summary="fallback", suggestions=["safe"])


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_valid_json_first_try(scripted):
    r = scripted(VALID).complete_json("sys", "user", S.ResumeSuggestions)
    assert r.data.summary == "ok" and r.attempts == 1 and not r.used_fallback


def test_retry_then_success(scripted):
    r = scripted("not json", '{"summary": "x"}', VALID).complete_json("s", "u", S.ResumeSuggestions)
    assert r.attempts == 3 and r.data.suggestions == ["Do X"] and not r.used_fallback


def test_max_two_retries_then_fallback(scripted):
    c = scripted("garbage")
    r = c.complete_json("s", "u", S.ResumeSuggestions, fallback=FALLBACK)
    assert ScriptedProvider.calls == 3  # 1 attempt + 2 retries, never more
    assert r.used_fallback and r.data.summary == "fallback" and r.error


def test_no_fallback_raises(scripted):
    with pytest.raises(LLMError):
        scripted("garbage").complete_json("s", "u", S.ResumeSuggestions)


def test_provider_error_is_retried_and_falls_back(scripted):
    c = scripted(LLMError("boom"), LLMError("boom"), LLMError("boom"))
    r = c.complete_json("s", "u", S.ResumeSuggestions, fallback=lambda: FALLBACK)
    assert r.used_fallback and ScriptedProvider.calls == 3


def test_missing_api_key_does_not_retry(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    c = LLMClient(provider="gemini", settings=get_settings())
    r = c.complete_json("s", "u", S.ResumeSuggestions, fallback=FALLBACK)
    assert r.used_fallback and "GEMINI_API_KEY" in (r.error or "")


def test_unknown_provider():
    with pytest.raises(LLMError):
        LLMClient(provider="nope")


def test_retry_message_includes_validation_error(scripted):
    c = scripted('{"summary": "x"}', VALID)
    seen = []
    orig = ScriptedProvider.generate

    def spy(self, system, user, *a, **k):
        seen.append(user)
        return orig(self, system, user, *a, **k)
    ScriptedProvider.generate = spy
    try:
        c.complete_json("s", "u", S.ResumeSuggestions)
    finally:
        ScriptedProvider.generate = orig
    assert "rejected" in seen[1] and "suggestions" in seen[1]


def test_question_schema_requires_six_tests_and_three_hints():
    base = dict(title="T", statement="x" * 30, input_format="i", output_format="o",
                examples=[{"input": "1", "output": "1"}], expected_time_complexity="O(n)",
                expected_space_complexity="O(1)", reference_solution="print(1)" * 3)
    tests = [{"input": str(i), "output": str(i)} for i in range(6)]
    ok = S.GeneratedQuestion(**base, test_cases=tests, hints=["a", "b", "c"])
    assert sum(not t.hidden for t in ok.test_cases) >= 1 and sum(t.hidden for t in ok.test_cases) >= 2
    with pytest.raises(ValidationError):
        S.GeneratedQuestion(**base, test_cases=tests[:5], hints=["a", "b", "c"])
    with pytest.raises(ValidationError):
        S.GeneratedQuestion(**base, test_cases=tests, hints=["a", "b"])


def test_rubric_scores_bounded():
    item = {"score": 11, "reason": "r"}
    with pytest.raises(ValidationError):
        S.RubricEvaluation(clarity=item, structure=item, relevance=item, confidence=item, grammar_vocabulary=item, depth=item)


def test_every_prompt_file_loads_and_renders():
    from backend.routers.prompts import SAMPLES

    prompts = list_prompts()
    assert len(prompts) >= 15
    for name in prompts:
        tpl = load_prompt(name)
        assert tpl.output_schema in S.SCHEMA_REGISTRY, name
        system, user = tpl.render(**SAMPLES[name])  # StrictUndefined: a missing sample variable fails here
        assert user.strip()
    assert load_prompt("rubric_evaluation", 1).version == 1 and load_prompt("rubric_evaluation").version == 2


def test_temperatures_follow_policy():
    assert load_prompt("rubric_evaluation").temperature == 0.2
    assert load_prompt("code_review").temperature == 0.2
    assert load_prompt("dsa_question_generation").temperature == 0.7
    assert load_prompt("soft_question_generation").temperature == 0.7
