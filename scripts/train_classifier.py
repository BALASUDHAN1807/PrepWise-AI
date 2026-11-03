"""Train and evaluate the soft-skill answer classifier (TF-IDF + structural features + Logistic Regression).

Usage:  python scripts/train_classifier.py            # trains, saves models/softskill_tfidf_lr.joblib, writes reports/
Reports three protocols (random CV, leave-questions-out, hold-out) and the length-only baseline, because on this
small synthetic dataset answer LENGTH alone predicts the label well - see docs/responsible_ai.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import classifier_service as C  # noqa: E402


def main() -> None:
    ev = C.evaluate()
    C.train(save=True)
    rows = ["| variant | random CV acc (mean±sd) | leave-questions-out acc | macro-F1 (LQO) |", "|---|---|---|---|"]
    for name, v in ev["variants"].items():
        r, lq = v["random_cv"], v["leave_questions_out"]
        rows.append(f"| {name} | {r.get('accuracy_mean', r.get('accuracy', '-'))} ± {r.get('accuracy_std', '-')} | "
                    f"{lq['accuracy']} | {lq['f1_macro']} |")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "classifier_eval.json").write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    md = ("# Soft-skill classifier evaluation\n\n" + "\n".join(rows) + f"\n\nHold-out (selected model): {ev['holdout_selected']}\n\n"
          f"> {ev['note']}\n")
    (out / "classifier_eval.md").write_text(md, encoding="utf-8")
    print(md)
    print("Model saved to", C.get_settings().models_path / C.MODEL_FILE)


if __name__ == "__main__":
    main()
