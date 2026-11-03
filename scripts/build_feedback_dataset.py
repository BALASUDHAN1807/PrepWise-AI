"""Build data/feedback_sft.jsonl - a SYNTHETIC supervised dataset for LoRA feedback generation (Unit 3/4).

Each row pairs an answer from data/soft_skill_labeled.csv with a short structured feedback paragraph:

    Verdict: <label>. Evidence: <words; STAR parts found; measurable detail?>. Strength: <...>. Improve <criterion>: <...>. Next step: <...>.

The feedback is composed DETERMINISTICALLY from (a) the human-style label in the CSV and (b) the transparent
rule-based rubric (backend/services/heuristics.py), so every claim in it is checkable against measurable features
of the answer (STAR parts, numbers, hedges, fillers, length). No LLM is involved and nothing is copied from the web.

Honest limits - read data/DATASET_CARD.md:
  * the text is templated, so a fine-tuned model can score well by learning the templates; metrics on this data
    measure format and groundedness, NOT pedagogical quality;
  * every row has ``reviewed: false``. Set it to true for rows a human has checked and train with
    ``--only-reviewed`` to use just those.

Usage:  python scripts/build_feedback_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _feedback_eval import compose  # noqa: E402


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "soft_skill_labeled.csv")
    out = ROOT / "data" / "feedback_sft.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in df.itertuples():
            row = {"id": int(r.id), "question_id": r.question_id, "category": r.category, "question": r.question,
                   "answer": r.answer, "label": r.label, "feedback": compose(r.question, r.answer, r.label),
                   "reviewed": False, "source": "template-composed from label + rule-based rubric; not LLM; not human-reviewed"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
