"""Shared helpers for the prompt / model / consistency experiments (Unit 3 & 4 evidence).

Task used by all three experiments: LLM rubric scoring of soft-skill answers, judged against the
poor/average/good labels in ``data/soft_skill_labeled.csv``.

IMPORTANT (read before quoting numbers): those labels were authored by an LLM-assisted script and have
not been human-reviewed, and the sample is small. The experiments therefore compare *strategies against
each other under identical conditions*; they do not measure agreement with real interviewers.
With ``LLM_PROVIDER=mock`` every strategy returns the same rule-based answer, so differences are zero by
construction - the scripts say so in their output.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.llm.client import LLMClient, LLMError  # noqa: E402
from backend.llm.prompt_loader import load_prompt  # noqa: E402
from backend.models.llm_schemas import CoTRubricEvaluation, RubricEvaluation  # noqa: E402

LABEL_ORDER = {"poor": 0, "average": 1, "good": 2}
REPORTS = ROOT / "reports"


def load_sample(n_per_label: int = 8, seed: int = 7) -> pd.DataFrame:
    """Stratified sample of labelled answers (same seed -> same sample for every strategy/model)."""
    df = pd.read_csv(ROOT / "data" / "soft_skill_labeled.csv")
    parts = [g.sample(min(n_per_label, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------- strategies --
ZERO_SHOT_SYSTEM = (
    "You grade interview answers. Rate the answer on clarity, structure, relevance, confidence, "
    "grammar_vocabulary and depth (integers 0-10, each with a short reason), detect the STAR parts and "
    "add a one-sentence overall_comment."
)
FEW_SHOT_EXAMPLES = (
    "Examples of graded answers (scores are clarity/structure/relevance/confidence/grammar_vocabulary/depth):\n"
    "Q: Tell me about a time you handled a conflict in a team.\n"
    "A: \"I don't really have conflicts, I get along with everyone.\" -> 5/1/2/4/7/1\n"
    "A: \"In my 3rd-year project two teammates disagreed on MongoDB vs MySQL, which blocked us for a week. As lead I ran a "
    "call, listed requirements and we built a small benchmark. MySQL fit our relational data, both agreed, and we delivered "
    "4 days early. I learned to move debates to data.\" -> 9/9/9/8/8/8\n"
    "Q: Why should we hire you?\nA: \"I work hard.\" -> 4/1/3/3/6/1\n\n"
)
ROLE_SYSTEM = (
    "You are a strict, fair HR interviewer for campus placements with ten years of experience. Judge content only; ignore "
    "accent, name, gender, college or region. Score the answer on clarity, structure (STAR), relevance, confidence, "
    "grammar_vocabulary and depth - integers 0-10 each with a one-sentence reason - and detect the STAR parts."
)
COT_SYSTEM = (
    "You are a strict, fair HR interviewer. First write your step-by-step reasoning in `reasoning` "
    "(what the answer does well, what is missing, evidence quotes). Then fill `evaluation` with six "
    "integer scores 0-10 (clarity, structure, relevance, confidence, grammar_vocabulary, depth) each with "
    "a one-sentence reason, plus star and overall_comment. Judge content only."
)
USER_QA = "Question: {q}\nAnswer: {a}"

STRATEGIES = {
    "zero_shot": "Minimal instruction: criteria names only, no persona, no examples",
    "few_shot": "Same minimal instruction plus three graded example answers",
    "role_based": "Persona (experienced HR interviewer) + criteria + bias guard (versioned prompt-file style, no examples)",
    "chain_of_thought": "Persona and criteria, but the model must write its reasoning before scoring (CoTRubricEvaluation schema)",
    "structured_output": "App default (rubric_evaluation v2): score anchors, calibration examples and strict Pydantic JSON schema",
}


def build_prompt(strategy: str, question: str, answer: str) -> tuple[str, str, type]:
    """Return (system, user, schema) for a strategy. All strategies must return JSON that validates against a schema."""
    qa = USER_QA.format(q=question, a=answer)
    if strategy == "zero_shot":
        return ZERO_SHOT_SYSTEM, qa, RubricEvaluation
    if strategy == "few_shot":
        return ZERO_SHOT_SYSTEM, FEW_SHOT_EXAMPLES + qa, RubricEvaluation
    if strategy == "role_based":
        return ROLE_SYSTEM, qa, RubricEvaluation
    if strategy == "chain_of_thought":
        return COT_SYSTEM, qa, CoTRubricEvaluation
    system, user = load_prompt("rubric_evaluation", 2).render(question=question, answer=answer)
    return system, user, RubricEvaluation


def score_one(client: LLMClient, strategy: str, row, temperature: float) -> dict:
    """Run one strategy on one answer. Never raises: invalid output is recorded, not hidden."""
    system, user, schema = build_prompt(strategy, row.question, row.answer)
    task = "exp_rubric_cot" if strategy == "chain_of_thought" else f"exp_rubric_{strategy}"
    t0 = time.perf_counter()
    out = {"id": int(row.id), "label": row.label, "strategy": strategy, "valid": False, "first_try": False,
           "attempts": 0, "total100": None, "error": None}
    try:
        res = client.complete_json(system, user, schema, temperature=temperature, task=task,
                                   variables={"question": row.question, "answer": row.answer})
        ev = res.data.evaluation if isinstance(res.data, CoTRubricEvaluation) else res.data
        out.update(valid=True, first_try=res.attempts == 1, attempts=res.attempts,
                   total100=round(ev.total() / 60 * 100, 1))
    except (LLMError, ValueError) as exc:
        out["error"] = str(exc)[:200]
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def band(score100: float) -> str:
    """Nearest of the app's classifier anchor points (poor 20 / average 55 / good 90)."""
    return "poor" if score100 < 37.5 else "average" if score100 < 72.5 else "good"


def summarize(rows: list[dict]) -> dict:
    """Aggregate metrics for one strategy/model."""
    ok = [r for r in rows if r["valid"]]
    out = {"n": len(rows), "valid_rate": round(len(ok) / len(rows), 3) if rows else 0,
           "first_try_valid_rate": round(sum(r["first_try"] for r in rows) / len(rows), 3) if rows else 0,
           "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in rows), 1) if rows else None}
    if len(ok) >= 3:
        y = [LABEL_ORDER[r["label"]] for r in ok]
        s = [r["total100"] for r in ok]
        rho = spearmanr(s, y).statistic if len(set(s)) > 1 else float("nan")
        out["score_mean"] = round(statistics.mean(s), 1)
        out["score_sd"] = round(statistics.pstdev(s), 1)
        out["spearman_vs_label"] = None if rho != rho else round(float(rho), 3)
        out["band_accuracy"] = round(sum(band(r["total100"]) == r["label"] for r in ok) / len(ok), 3)
        out["mean_score_by_label"] = {lab: round(statistics.mean(r["total100"] for r in ok if r["label"] == lab), 1)
                                      for lab in LABEL_ORDER if any(r["label"] == lab for r in ok)}
    return out


def markdown_table(summaries: dict[str, dict], cols: list[str]) -> str:
    head = "| name | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = ""
    for name, s in summaries.items():
        body += f"| {name} | " + " | ".join(str(s.get(c, "-")) for c in cols) + " |\n"
    return head + body


def mock_banner(provider: str) -> str:
    if provider != "mock":
        return ""
    return ("> **Mock provider in use.** All strategies/models return the same rule-based heuristic, so the "
            "comparison below is degenerate (identical by construction). Set `LLM_PROVIDER` and an API key "
            "in `.env` and re-run to get a real comparison.\n\n")


def write_report(name: str, title: str, body_md: str, payload: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    (REPORTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    path = REPORTS / f"{name}.md"
    path.write_text(f"# {title}\n\n_Generated {stamp} by `scripts/{name}.py`_\n\n{body_md}", encoding="utf-8")
    return path


def make_client(provider: str | None, model: str | None = None) -> LLMClient:
    return LLMClient(provider or get_settings().llm_provider, model)
