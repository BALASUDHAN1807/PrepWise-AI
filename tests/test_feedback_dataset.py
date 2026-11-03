"""The synthetic feedback dataset and the groundedness metrics that audit generated feedback."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _feedback_eval as E  # noqa: E402

ROWS = [json.loads(line) for line in (ROOT / "data" / "feedback_sft.jsonl").read_text(encoding="utf-8").splitlines()]


def test_dataset_shape_and_honesty_flags():
    assert len(ROWS) >= 150
    assert {r["label"] for r in ROWS} == {"poor", "average", "good"}
    assert all(r["reviewed"] is False for r in ROWS)           # nothing is claimed to be human-reviewed
    assert all("not human-reviewed" in r["source"] for r in ROWS)


def test_references_pass_their_own_audit_and_parser_roundtrips():
    res = [E.score_feedback(r["feedback"], r["question"], r["answer"], r["label"], r["feedback"]) for r in ROWS]
    agg = E.aggregate(res)
    assert agg["format_ok"] == agg["verdict_correct"] == agg["evidence_star_ok"] == agg["improve_is_weakest"] == 1.0
    assert agg["rouge_l"] == 1.0


def test_audit_catches_false_claims():
    r = next(x for x in ROWS if x["label"] == "poor")
    wrong = r["feedback"].replace("Verdict: poor", "Verdict: good")
    assert not E.score_feedback(wrong, r["question"], r["answer"], r["label"], r["feedback"])["verdict_correct"]
    lying = r["feedback"].replace("no measurable detail", "includes measurable detail")
    assert not E.score_feedback(lying, r["question"], r["answer"], r["label"], r["feedback"])["evidence_number_ok"]
    assert not E.score_feedback("Great answer!", r["question"], r["answer"], r["label"], r["feedback"])["format_ok"]


def test_rouge_l_basics():
    assert E.rouge_l("a b c d", "a b c d") == 1.0
    assert E.rouge_l("x y", "a b") == 0.0
    assert 0 < E.rouge_l("a b c", "a x c") < 1


def test_free_text_improve_line_is_mapped_to_a_criterion():
    p = E.parse_feedback("Verdict: poor. Strength: ok. Improve: add a specific example with numbers. Next step: practise.")
    assert p["improve_criterion"] == "depth" and p["format_ok"]
