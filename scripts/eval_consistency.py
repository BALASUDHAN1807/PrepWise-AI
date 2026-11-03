"""Measure evaluator consistency: run the SAME answer through the rubric evaluator several times - Unit 3/5.

Default (as in the course spec): one answer x 5 runs -> mean and standard deviation of the 0-100 score.
Add ``--n 6`` to repeat the exercise for several answers (spread evenly across poor/average/good), and
``--temperatures 0.0,0.2,0.7`` to see the effect of sampling temperature (the app evaluates at 0.2).

Usage:  python scripts/eval_consistency.py [--provider gemini] [--repeats 5] [--n 1] [--temperatures 0.2]
Writes reports/eval_consistency.{md,json}. A deterministic provider (mock) shows zero spread by design.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from _eval_common import band, load_sample, make_client, markdown_table, mock_banner, score_one, write_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--strategy", default="structured_output")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n", type=int, default=1, help="number of different answers (taken evenly across labels)")
    ap.add_argument("--temperatures", default="0.2", help="comma list; the app uses 0.2 for evaluation")
    args = ap.parse_args()

    client = make_client(args.provider, args.model)
    sample = load_sample(max(1, -(-args.n // 3))).head(args.n) if args.n > 1 else load_sample(1).query("label == 'average'").head(1)
    per_answer, summary = [], {}
    for temp in [float(t) for t in args.temperatures.split(",")]:
        sds, ranges, agree, valid, first, total = [], [], [], 0, 0, 0
        for row in sample.itertuples():
            runs = [score_one(client, args.strategy, row, temp) for _ in range(args.repeats)]
            total += len(runs)
            valid += sum(r["valid"] for r in runs)
            first += sum(r["first_try"] for r in runs)
            scores = [r["total100"] for r in runs if r["valid"]]
            if scores:
                sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
                sds.append(sd)
                ranges.append(max(scores) - min(scores))
                agree.append(Counter(band(s) for s in scores).most_common(1)[0][1] / len(scores))
                per_answer.append({"answer_id": int(row.id), "label": row.label, "temperature": temp, "runs": scores,
                                   "mean": round(statistics.mean(scores), 2), "sd": round(sd, 2)})
        summary[f"T={temp}"] = {
            "mean_sd_across_answers": round(statistics.mean(sds), 2) if sds else "-",
            "mean_range": round(statistics.mean(ranges), 2) if ranges else "-",
            "same_band_rate": round(statistics.mean(agree), 3) if agree else "-",
            "valid_rate": round(valid / total, 3), "first_try_valid_rate": round(first / total, 3)}
        print(f"T={temp}: {summary[f'T={temp}']}")
    for r in per_answer:
        print(f"  answer {r['answer_id']} ({r['label']}) T={r['temperature']}: scores {r['runs']} -> mean {r['mean']}, sd {r['sd']}")

    detail = "| answer | label | T | scores | mean | sd |\n|---|---|---|---|---|---|\n" + "\n".join(
        f"| {r['answer_id']} | {r['label']} | {r['temperature']} | {', '.join(map(str, r['runs']))} | {r['mean']} | {r['sd']} |"
        for r in per_answer)
    body = (mock_banner(client.provider_name)
            + f"Provider `{client.provider_name}/{client.model}`, strategy `{args.strategy}`, {len(sample)} answer(s) x "
              f"{args.repeats} runs.\n\n**Per answer**\n\n{detail}\n\n**Summary**\n\n"
            + markdown_table(summary, ["mean_sd_across_answers", "mean_range", "same_band_rate", "valid_rate", "first_try_valid_rate"])
            + "\n**Why it matters.** A rubric score that swings by 10+ points between identical runs should not be shown to a "
              "student as if it were precise. The UI therefore labels scores as guidance, evaluation runs at temperature 0.2, "
              "and the DSA correctness score never depends on an LLM.\n")
    print("Report:", write_report("eval_consistency", "Evaluation consistency", body, {"summary": summary, "per_answer": per_answer}))


if __name__ == "__main__":
    main()
