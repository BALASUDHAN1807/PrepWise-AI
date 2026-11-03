"""Template bank, parser and groundedness metrics for the synthetic feedback task.

The reference feedback is composed deterministically (``compose``) from measurable features of the answer, which lets
us check any generator's output - prompting, RAG or a LoRA model - against facts instead of vibes:

    Verdict: <poor|average|good>. Evidence: <N> words; STAR parts found: <list|none>; <includes|no> measurable detail.
    Strength: <sentence>. Improve <criterion>: <sentence>. Next step: <sentence>.

Metrics (see ``score_feedback``): format validity, verdict accuracy, evidence accuracy (STAR set + number flag),
whether the criterion named under "Improve" is really the weakest one, whether the claimed strength is supported, and
ROUGE-L against the reference. Because the reference is templated, ROUGE-L rewards models that learned the template;
the groundedness metrics are the more meaningful ones - and still only as good as the rule-based rubric behind them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import heuristics as H  # noqa: E402

CRITERIA = ["clarity", "structure", "relevance", "confidence", "grammar_vocabulary", "depth"]
LABELS = ["poor", "average", "good"]
STAR_PARTS = ["situation", "task", "action", "result"]

STRENGTH = {
    "clarity": "The answer is easy to follow.",
    "structure": "The answer follows a clear order (situation, action, outcome).",
    "relevance": "You answer the question that was asked.",
    "confidence": "You take ownership with clear 'I' statements.",
    "grammar_vocabulary": "The language is correct and professional.",
    "depth": "You include specific detail such as numbers or lessons learned.",
}
IMPROVE = {
    "clarity": "use shorter sentences and cut filler words.",
    "structure": "follow STAR and say what the situation, your task, your action and the result were.",
    "relevance": "address the exact question and reuse its key terms.",
    "confidence": "say what you did, and avoid hedges like 'maybe' or 'I think'.",
    "grammar_vocabulary": "start sentences with a capital letter, end them properly and use precise words.",
    "depth": "add a specific example, a number and what you learned.",
}
NEXT = {
    "poor": "rewrite the answer in four sentences, one each for situation, task, action and result.",
    "average": "add one measurable result and one lesson you learned.",
    "good": "keep it under two minutes and lead with the result.",
}
NO_STRENGTH = "No single strength stands out yet; the attempt to answer is a start."

# Free-text hints so LLM output that does not copy our wording can still be mapped to a criterion.
CRITERION_WORDS = {
    "clarity": ["clarity", "clear", "concise", "shorter sentence", "rambl", "filler"],
    "structure": ["structure", "star", "organis", "organiz", "logical order", "situation"],
    "relevance": ["relevance", "relevant", "on topic", "exact question", "addresses the question", "answer the question"],
    "confidence": ["confidence", "confident", "ownership", "hedg", "assertive"],
    "grammar_vocabulary": ["grammar", "vocabulary", "language", "capital", "punctuation"],
    "depth": ["depth", "specific", "example", "measurable", "number", "detail", "quantif"],
}


def answer_features(question: str, answer: str) -> dict:
    """Ground truth used both to compose references and to audit generated feedback."""
    ev = H.heuristic_rubric(question, answer)
    scores = {c: getattr(ev, c).score for c in CRITERIA}
    star = H.detect_star(answer)
    return {"scores": scores,
            "best": max(CRITERIA, key=lambda c: (scores[c], -CRITERIA.index(c))),
            "weakest": min(CRITERIA, key=lambda c: (scores[c], CRITERIA.index(c))),
            "words": len(H.words(answer)),
            "star": [p for p in STAR_PARTS if getattr(star, p)],
            "has_number": bool(H.NUMBER_RE.search(answer))}


def compose(question: str, answer: str, label: str) -> str:
    """Reference feedback for one answer."""
    f = answer_features(question, answer)
    star = ", ".join(f["star"]) or "none"
    strength = STRENGTH[f["best"]] if f["scores"][f["best"]] >= 7 else NO_STRENGTH
    return (f"Verdict: {label}. Evidence: {f['words']} words; STAR parts found: {star}; "
            f"{'includes' if f['has_number'] else 'no'} measurable detail. Strength: {strength} "
            f"Improve {f['weakest']}: {IMPROVE[f['weakest']]} Next step: {NEXT[label]}")


# ---------------------------------------------------------------- parsing ----
def _criterion_from_text(text: str) -> str | None:
    low = text.lower()
    hits = {c: sum(w in low for w in ws) for c, ws in CRITERION_WORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] else None


def parse_feedback(text: str) -> dict:
    """Pull the structured fields out of generated text. Missing fields are None; ``format_ok`` needs all four sections."""
    t = " ".join((text or "").split())
    verdict = re.search(r"verdict:\s*(poor|average|good)", t, re.I)
    ev = re.search(r"evidence:\s*(\d+)\s*words?;\s*STAR parts found:\s*([a-z, ]+?);\s*(includes|no) measurable", t, re.I)
    strength = re.search(r"strength:\s*(.*?)\s*(?:improve|next step:|$)", t, re.I)
    improve = re.search(r"improve\s*([^:]*):\s*(.*?)\s*(?:next step:|$)", t, re.I)
    nxt = re.search(r"next step:\s*(.+)$", t, re.I)
    named = (improve.group(1).strip().lower().replace(" ", "_").replace("&", "").replace("/", "_") if improve else "")
    criterion = named if named in CRITERIA else (_criterion_from_text(improve.group(1) + " " + improve.group(2)) if improve else None)
    star = None
    if ev:
        star = [] if ev.group(2).strip().lower() == "none" else [s.strip().lower() for s in ev.group(2).split(",")]
    return {"verdict": verdict.group(1).lower() if verdict else None,
            "evidence_words": int(ev.group(1)) if ev else None, "evidence_star": star,
            "evidence_number": (ev.group(3).lower() == "includes") if ev else None,
            "strength": strength.group(1) if strength else None,
            "improve_criterion": criterion, "has_next": bool(nxt),
            "format_ok": bool(verdict and strength and improve and nxt)}


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", ref.lower())
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def strength_supported(strength: str | None, feats: dict) -> bool | None:
    """True if the claimed strength's criterion scores >= 6 on the rule-based rubric (None if no claim is made)."""
    if not strength or strength.strip().lower().startswith("no single strength"):
        return None
    crit = _criterion_from_text(strength)
    return None if crit is None else feats["scores"][crit] >= 6


def score_feedback(pred: str, question: str, answer: str, label: str, reference: str) -> dict:
    """Per-example checks of one generated feedback against measurable facts."""
    p, f = parse_feedback(pred), answer_features(question, answer)
    sup = strength_supported(p["strength"], f)
    return {"format_ok": p["format_ok"], "verdict_correct": p["verdict"] == label,
            "evidence_star_ok": p["evidence_star"] is not None and set(p["evidence_star"]) == set(f["star"]),
            "evidence_number_ok": p["evidence_number"] is not None and p["evidence_number"] == f["has_number"],
            "improve_is_weakest": p["improve_criterion"] == f["weakest"],
            "strength_supported": sup, "rouge_l": rouge_l(pred, reference)}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in ("format_ok", "verdict_correct", "evidence_star_ok", "evidence_number_ok", "improve_is_weakest"):
        out[k] = round(sum(r[k] for r in rows) / n, 3) if n else 0
    claimed = [r["strength_supported"] for r in rows if r["strength_supported"] is not None]
    out["strength_supported"] = round(sum(claimed) / len(claimed), 3) if claimed else "-"
    out["rouge_l"] = round(sum(r["rouge_l"] for r in rows) / n, 3) if n else 0
    return out
