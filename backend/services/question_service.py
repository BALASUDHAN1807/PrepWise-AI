"""DSA / SQL question generation with RAG grounding and a hallucination guard.

Pipeline (fixed, deterministic - no agent loop):

1. RAG: retrieve similar questions from ``data/question_bank.json`` (style grounding).
2. Prompt: ask the LLM for an ORIGINAL question as JSON (validated by Pydantic).
3. Guard: run the LLM's *reference solution* against ALL of its test cases in the sandbox.
   If any case fails, the question is rejected and regenerated (max ``MAX_GENERATION_ATTEMPTS``).
4. If every attempt fails (or the LLM is unavailable) a verified bank question is served and
   flagged ``source="bank"``. Bank questions pass through the same guard in the test-suite.
"""
from __future__ import annotations

import logging

from backend.llm.client import LLMClient
from backend.llm.runner import run_prompt
from backend.models.llm_schemas import GeneratedQuestion
from backend.services import rag_service as R
from backend.services.code_runner import LocalSandbox, get_sandbox

log = logging.getLogger("prepwise.question")

MAX_GENERATION_ATTEMPTS = 3
TOPICS = ["arrays", "strings", "linked_lists", "trees", "graphs", "dp", "sql"]
DIFFICULTIES = ["easy", "medium", "hard"]


class QuestionGuardError(RuntimeError):
    pass


def guard_question(q: GeneratedQuestion | dict, sandbox=None) -> dict:
    """Execute the reference solution on every test case.

    Returns ``{"ok": bool, "passed": n, "total": m, "reason": str}``. Structural checks
    (enough tests, distinct inputs, hidden/visible mix, non-empty outputs) come first.
    """
    data = q.model_dump() if isinstance(q, GeneratedQuestion) else q
    tests = data["test_cases"]
    kind = data.get("kind", "code")
    if len(tests) < 6:
        return {"ok": False, "passed": 0, "total": len(tests), "reason": "fewer than 6 test cases"}
    if len({t["input"] for t in tests}) < len(tests):
        return {"ok": False, "passed": 0, "total": len(tests), "reason": "duplicate test inputs"}
    if kind == "code" and any(not t["output"].strip() for t in tests):
        return {"ok": False, "passed": 0, "total": len(tests), "reason": "a test case has empty expected output"}
    sb = sandbox or (LocalSandbox() if kind == "sql" else get_sandbox())
    lang = "sql" if kind == "sql" else "python"
    try:
        results = sb.run_tests(lang, data["reference_solution"], tests, kind)
    except Exception as exc:  # unsupported language, sandbox failure
        return {"ok": False, "passed": 0, "total": len(tests), "reason": f"sandbox error: {type(exc).__name__}"}
    passed = sum(r.passed for r in results)
    first_fail = next((r for r in results if not r.passed), None)
    reason = "" if first_fail is None else f"reference solution failed case {first_fail.index}: {first_fail.status}"
    return {"ok": passed == len(tests), "passed": passed, "total": len(tests), "reason": reason}


def _strip(q: dict) -> dict:
    return {k: q[k] for k in GeneratedQuestion.model_fields if k in q}


def bank_question(topic: str, difficulty: str, kind: str, avoid_titles: list[str] | None = None,
                  company_style: str | None = None) -> dict:
    """Pick a curated bank question matching the request as closely as possible."""
    bank = R.load_question_bank()
    avoid = {t.lower() for t in (avoid_titles or [])}

    def pick(pool):
        fresh = [q for q in pool if q["title"].lower() not in avoid]
        return (fresh or pool or [None])[0]

    same_kind = [q for q in bank if q.get("kind", "code") == kind]
    for pool in (
        [q for q in same_kind if q["topic"] == topic and q["difficulty"] == difficulty
         and (not company_style or company_style in q.get("company_styles", []))],
        [q for q in same_kind if q["topic"] == topic and q["difficulty"] == difficulty],
        [q for q in same_kind if q["topic"] == topic],
        [q for q in same_kind if q["difficulty"] == difficulty],
        same_kind,
    ):
        choice = pick(pool)
        if choice:
            return _strip(choice)
    raise QuestionGuardError("The question bank is empty")


def generate_question(
    topic: str,
    difficulty: str,
    role: str = "Software Engineer",
    company_style: str = "product",
    skills: list[str] | None = None,
    weak_areas: list[str] | None = None,
    avoid_titles: list[str] | None = None,
    client: LLMClient | None = None,
    sandbox=None,
) -> dict:
    """Return ``{"question": dict, "meta": {...}}``; the question includes its reference solution
    (callers must strip it before sending to the browser - see :func:`public_view`)."""
    if topic not in TOPICS:
        raise ValueError(f"Unknown topic '{topic}'. Choose one of {TOPICS}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose one of {DIFFICULTIES}")
    kind = "sql" if topic == "sql" else "code"
    skills, weak_areas, avoid_titles = skills or [], weak_areas or [], avoid_titles or []

    rag_hits = R.retrieve_questions(f"{topic} {difficulty} {role} {' '.join(weak_areas)}", k=3, topic=topic, kind=kind)
    examples = "\n\n".join(
        f"### {h['title']} ({h['difficulty']})\n{h['statement']}\nConstraints: {'; '.join(h['constraints'])}\n"
        f"Input format: {h['input_format']}\nOutput format: {h['output_format']}"
        for h in rag_hits) or "(no examples available)"

    meta = {"attempts": 0, "rag_examples": [h["title"] for h in rag_hits], "rejections": [], "source": "llm",
            "embedding": R.embedding_info()}
    feedback_note = ""
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        meta["attempts"] = attempt
        ctx = {"topic": topic, "difficulty": difficulty, "kind": kind, "avoid_titles": avoid_titles}
        # No fallback here on purpose: an unusable LLM raises LLMError, which
        # safe_generate_question() turns into a verified bank question.
        res = run_prompt(
            "dsa_question_generation",
            {"topic": topic, "difficulty": difficulty, "role": role, "company_style": company_style, "kind": kind,
             "skills": skills, "weak_areas": weak_areas, "avoid_titles": avoid_titles, "examples": examples + feedback_note,
             "_ctx": ctx},
            client=client)
        q = res.data
        q.kind = kind
        q.topic = topic if not q.topic else q.topic
        q.difficulty = difficulty
        guard = guard_question(q, sandbox)
        meta["guard"] = guard
        meta["llm"] = res.info()
        if guard["ok"]:
            return {"question": q.model_dump(), "meta": meta}
        meta["rejections"].append(guard["reason"])
        feedback_note = (f"\n\nNOTE: your previous attempt was rejected because the {guard['reason']}. "
                         "Make sure the reference solution prints EXACTLY the expected output for every test case.")
        log.info("question rejected by guard (attempt %d): %s", attempt, guard["reason"])
    return _serve_bank(topic, difficulty, kind, avoid_titles, company_style, meta, sandbox)


def _serve_bank(topic, difficulty, kind, avoid_titles, company_style, meta, sandbox) -> dict:
    q = bank_question(topic, difficulty, kind, avoid_titles, company_style)
    meta["source"] = "bank"
    meta["guard"] = guard_question(q, sandbox)
    return {"question": q, "meta": meta}


def safe_generate_question(*args, **kwargs) -> dict:
    """Like :func:`generate_question` but serves a verified bank question on any LLM failure."""
    from backend.llm.client import LLMError

    try:
        return generate_question(*args, **kwargs)
    except LLMError as exc:
        kw = dict(kwargs)
        topic = kw.get("topic") or args[0]
        difficulty = kw.get("difficulty") or args[1]
        kind = "sql" if topic == "sql" else "code"
        meta = {"attempts": 0, "rag_examples": [], "rejections": [], "llm_error": str(exc)[:200]}
        return _serve_bank(topic, difficulty, kind, kw.get("avoid_titles"), kw.get("company_style"), meta, kw.get("sandbox"))


def public_view(q: dict) -> dict:
    """Question as shown to the candidate: hidden test data and the reference solution are removed."""
    out = {k: v for k, v in q.items() if k not in {"reference_solution", "hints", "test_cases"}}
    out["visible_tests"] = [t for t in q["test_cases"] if not t.get("hidden")]
    out["hidden_test_count"] = sum(1 for t in q["test_cases"] if t.get("hidden"))
    out["hint_count"] = len(q.get("hints", []))
    return out
