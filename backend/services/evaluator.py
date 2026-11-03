"""Scoring and evaluation.

DSA   : correctness comes ONLY from executing test cases in the sandbox. The LLM reviews
        logic / complexity / style and its review is capped by the test pass-rate.
Soft  : an LLM rubric evaluator (JSON, with reasons) runs side by side with the
        discriminative classifier (see ``classifier_service``).

All formulas are pure functions so they can be unit-tested and shown in the explanations panel.
"""
from __future__ import annotations

from backend.core.config import get_settings
from backend.llm.client import LLMClient
from backend.llm.runner import run_prompt
from backend.models import llm_schemas as S
from backend.services import heuristics as H
from backend.services.code_runner import CaseResult, get_sandbox

DSA_WEIGHTS = {"tests": 0.60, "logic": 0.20, "complexity": 0.10, "quality": 0.10}
SOFT_WEIGHTS = {"rubric": 0.70, "classifier": 0.20, "delivery": 0.10}
CLASSIFIER_POINTS = {"poor": 20.0, "average": 55.0, "good": 90.0}
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


# ----------------------------------------------------------------- DSA math --
def cap_review_by_tests(score_0_10: float, pass_rate: float, floor: float = 3.0) -> float:
    """An LLM review score cannot exceed ``floor + (10 - floor) * pass_rate``.

    Logic and complexity use floor=3 (code that does not solve the problem cannot be credited with an
    efficient or sound approach); code quality uses a gentler floor=5 because style can be judged on
    non-working code. With every test failing the three LLM components can add at most 6+3+5 = 14 points.
    """
    return min(float(score_0_10), floor + (10.0 - floor) * pass_rate)


def compute_dsa_score(pass_rate: float, logic: float, complexity: float, quality: float, hints_used: int,
                      hint_penalty: float | None = None) -> dict:
    """Score out of 100 = 60% tests + 20% logic + 10% complexity + 10% code quality - hint penalty.

    ``logic``, ``complexity`` and ``quality`` are 0-10 LLM review scores; ``pass_rate`` is 0-1.
    """
    penalty_each = get_settings().hint_penalty if hint_penalty is None else hint_penalty
    pass_rate = max(0.0, min(1.0, pass_rate))
    logic = cap_review_by_tests(logic, pass_rate)
    complexity = cap_review_by_tests(complexity, pass_rate)
    quality = cap_review_by_tests(quality, pass_rate, floor=5.0)
    parts = {
        "tests": round(DSA_WEIGHTS["tests"] * 100 * pass_rate, 2),
        "logic": round(DSA_WEIGHTS["logic"] * 10 * max(0, min(10, logic)), 2),
        "complexity": round(DSA_WEIGHTS["complexity"] * 10 * max(0, min(10, complexity)), 2),
        "quality": round(DSA_WEIGHTS["quality"] * 10 * max(0, min(10, quality)), 2),
    }
    penalty = round(penalty_each * max(0, hints_used), 2)
    total = max(0.0, min(100.0, sum(parts.values()) - penalty))
    return {"total": round(total, 1), "components": parts, "hint_penalty": penalty, "hints_used": hints_used,
            "weights": DSA_WEIGHTS, "logic_after_cap": round(logic, 2),
            "complexity_after_cap": round(complexity, 2), "quality_after_cap": round(quality, 2)}


def next_difficulty(current: str, last_score: float | None) -> str:
    """Adaptive difficulty: >= 80 moves up one level, < 40 moves down, otherwise stays."""
    i = DIFFICULTY_ORDER.index(current) if current in DIFFICULTY_ORDER else 1
    if last_score is None:
        return DIFFICULTY_ORDER[i]
    if last_score >= 80:
        i = min(i + 1, len(DIFFICULTY_ORDER) - 1)
    elif last_score < 40:
        i = max(i - 1, 0)
    return DIFFICULTY_ORDER[i]


def summarize_cases(results: list[CaseResult]) -> dict:
    passed = sum(r.passed for r in results)
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "visible_passed": sum(r.passed for r in results if not r.hidden),
        "visible_total": sum(1 for r in results if not r.hidden),
        "hidden_passed": sum(r.passed for r in results if r.hidden),
        "hidden_total": sum(1 for r in results if r.hidden),
        "max_runtime_ms": max((r.runtime_ms for r in results), default=0.0),
        "cases": [r.public() for r in results],
    }


def review_code(question: dict, language: str, code: str, summary: dict, client: LLMClient | None = None) -> tuple[S.CodeReview, dict]:
    failures = [f"case {c['index']}: {c['status']}" for c in summary["cases"] if c["status"] != "passed"][:6]
    ctx = {"code": code, "language": language, "passed": summary["passed"], "total": summary["total"],
           "expected_time": question["expected_time_complexity"]}
    res = run_prompt(
        "code_review",
        {"title": question["title"], "statement": question["statement"][:1500],
         "expected_time": question["expected_time_complexity"], "expected_space": question["expected_space_complexity"],
         "language": language, "passed": summary["passed"], "total": summary["total"], "failures": failures or "none",
         "code": code[:6000], "_ctx": ctx},
        fallback=lambda: H.heuristic_code_review(code, language, summary["passed"], summary["total"],
                                                 question["expected_time_complexity"]),
        client=client)
    return res.data, res.info()


def evaluate_submission(question: dict, language: str, code: str, hints_used: int = 0,
                        client: LLMClient | None = None, sandbox=None) -> dict:
    """Run all test cases in the sandbox, ask the LLM for a logic review, compute the 0-100 score."""
    sb = sandbox or get_sandbox()
    results = sb.run_tests(language if question.get("kind") != "sql" else "sql", code, question["test_cases"],
                           question.get("kind", "code"))
    summary = summarize_cases(results)
    review, info = review_code(question, language, code, summary, client)
    score = compute_dsa_score(summary["pass_rate"], review.approach_quality, review.complexity_score,
                              review.code_quality, hints_used)
    return {"tests": summary, "review": review.model_dump(), "review_llm": info, "score": score}


# ---------------------------------------------------------------- soft skills --
def rubric_evaluate(question: str, answer: str, client: LLMClient | None = None, version: int | None = None
                    ) -> tuple[S.RubricEvaluation, dict]:
    """LLM rubric evaluator: six criteria, each 0-10 with a reason (temperature 0.2)."""
    res = run_prompt(
        "rubric_evaluation", {"question": question, "answer": answer},
        fallback=lambda: H.heuristic_rubric(question, answer), client=client, version=version)
    return res.data, res.info()


def rubric_score_100(r: S.RubricEvaluation) -> float:
    return round(r.total() / 60 * 100, 1)


def delivery_score(analytics: dict) -> float:
    """0-100 from filler rate and (for voice) pace. Typed answers are judged on fillers only."""
    s = 100.0 - min(60.0, analytics.get("filler_rate_per_100_words", 0.0) * 6)
    if analytics.get("pace") in ("too slow", "too fast"):
        s -= 10
    pauses = analytics.get("pauses")
    if pauses and pauses.get("count", 0) > 5:
        s -= 10
    return round(max(0.0, s), 1)


def classifier_points(probabilities: dict[str, float] | None) -> float | None:
    """Expected score (0-100) implied by the classifier's class probabilities."""
    if not probabilities:
        return None
    return round(sum(CLASSIFIER_POINTS[k] * p for k, p in probabilities.items() if k in CLASSIFIER_POINTS), 1)


def compute_soft_score(rubric_100: float, classifier_proba: dict[str, float] | None, analytics: dict) -> dict:
    """Overall soft-skill score = 70% rubric + 20% classifier expectation + 10% delivery.

    If the classifier is unavailable its weight is redistributed to the rubric.
    """
    cls = classifier_points(classifier_proba)
    delivery = delivery_score(analytics)
    w = dict(SOFT_WEIGHTS)
    if cls is None:
        w["rubric"] += w.pop("classifier")
        cls = 0.0
        w["classifier"] = 0.0
    total = w["rubric"] * rubric_100 + w["classifier"] * cls + w["delivery"] * delivery
    return {"total": round(total, 1), "components": {"rubric": rubric_100, "classifier": cls if w["classifier"] else None,
                                                     "delivery": delivery}, "weights": w}


def followup_question(question: str, answer: str, depth: int, client: LLMClient | None = None) -> S.FollowUp:
    """Prompt-chain step: derive ONE follow-up from the last answer (max 2 per question, enforced in code)."""
    max_depth = get_settings().max_followups
    if depth >= max_depth:
        return S.FollowUp(needs_followup=False, reason="Maximum follow-ups reached.")
    res = run_prompt(
        "followup_question", {"question": question, "answer": answer, "depth": depth, "max_depth": max_depth},
        fallback=lambda: H.heuristic_followup(question, answer, depth, max_depth), client=client)
    fu = res.data
    if fu.needs_followup and not fu.question.strip():
        fu = S.FollowUp(needs_followup=False, reason="Model returned an empty follow-up.")
    return fu
