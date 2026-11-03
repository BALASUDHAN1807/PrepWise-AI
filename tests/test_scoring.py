"""Scoring formulas: DSA score, adaptive difficulty, soft-skill score, readiness."""
import pytest

from backend.services import evaluator as E
from backend.services import feedback_service as F


def test_dsa_weights_sum_to_one():
    assert sum(E.DSA_WEIGHTS.values()) == pytest.approx(1.0)
    assert E.DSA_WEIGHTS == {"tests": 0.6, "logic": 0.2, "complexity": 0.1, "quality": 0.1}


def test_perfect_score_is_100():
    s = E.compute_dsa_score(1.0, 10, 10, 10, 0, hint_penalty=5)
    assert s["total"] == 100.0 and s["components"] == {"tests": 60.0, "logic": 20.0, "complexity": 10.0, "quality": 10.0}


def test_known_mixed_score():
    # 0.5 pass -> 30; logic 8 (cap 3+3.5=6.5) -> 13; complexity 5 -> 5; quality 6 -> 6 => 54
    s = E.compute_dsa_score(0.5, 8, 5, 6, 0, hint_penalty=5)
    assert s["total"] == 54.0 and s["logic_after_cap"] == 6.5


def test_hint_penalty_is_applied_per_hint_and_floors_at_zero():
    base = E.compute_dsa_score(1.0, 10, 10, 10, 0, hint_penalty=5)["total"]
    assert E.compute_dsa_score(1.0, 10, 10, 10, 2, hint_penalty=5)["total"] == base - 10
    assert E.compute_dsa_score(0.0, 0, 0, 0, 3, hint_penalty=5)["total"] == 0.0


def test_llm_cannot_award_marks_to_code_that_fails_every_test():
    # LLM claims 10/10/10 for code failing all tests: logic cap 3 -> 6 pts, complexity cap 3 -> 3, quality cap 5 -> 5.
    s = E.compute_dsa_score(0.0, 10, 10, 10, 0, hint_penalty=5)
    c = s["components"]
    assert c == {"tests": 0.0, "logic": 6.0, "complexity": 3.0, "quality": 5.0} and s["total"] == 14.0


def test_caps_relax_as_tests_pass():
    assert E.cap_review_by_tests(10, 1.0) == 10 and E.cap_review_by_tests(10, 0.5) == 6.5
    assert E.cap_review_by_tests(2, 1.0) == 2          # caps never raise a low score
    assert E.cap_review_by_tests(10, 0.0, floor=5) == 5


def test_score_inputs_are_clamped():
    s = E.compute_dsa_score(1.7, 99, -4, 50, 0, hint_penalty=0)
    assert 0 <= s["total"] <= 100


@pytest.mark.parametrize("cur,score,expected", [
    ("easy", 90, "medium"), ("medium", 85, "hard"), ("hard", 95, "hard"),
    ("medium", 30, "easy"), ("easy", 10, "easy"), ("medium", 60, "medium"), ("medium", None, "medium")])
def test_adaptive_difficulty(cur, score, expected):
    assert E.next_difficulty(cur, score) == expected


def test_soft_score_weights_and_components():
    assert sum(E.SOFT_WEIGHTS.values()) == pytest.approx(1.0)
    analytics = {"filler_rate_per_100_words": 0.0}
    s = E.compute_soft_score(80.0, {"poor": 0.0, "average": 0.0, "good": 1.0}, analytics)
    # 0.7*80 + 0.2*90 + 0.1*100 = 84
    assert s["total"] == 84.0


def test_soft_score_without_classifier_redistributes_weight():
    s = E.compute_soft_score(80.0, None, {"filler_rate_per_100_words": 0.0})
    assert s["weights"]["classifier"] == 0 and s["total"] == pytest.approx(0.9 * 80 + 0.1 * 100)


def test_delivery_penalises_fillers_and_pace():
    clean = E.delivery_score({"filler_rate_per_100_words": 0})
    messy = E.delivery_score({"filler_rate_per_100_words": 10, "pace": "too fast"})
    assert clean == 100 and messy < clean - 20


def test_readiness_weighted_and_renormalised():
    w = {"resume": 0.2, "dsa": 0.4, "soft": 0.4}
    assert F.compute_readiness(100, 50, 0, w)["score"] == pytest.approx(40.0)
    only_dsa = F.compute_readiness(None, 70, None, w)
    assert only_dsa["score"] == 70.0 and only_dsa["weights"] == {"dsa": 1.0}
    assert F.compute_readiness(None, None, None, w)["score"] is None
