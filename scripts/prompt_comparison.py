"""Compare prompting strategies on the same answers: zero-shot, few-shot, role-based, chain-of-thought, structured-output - Unit 3.

Usage:  python scripts/prompt_comparison.py [--provider gemini] [--model gemini-3.8-flash]
                                              [--n-per-label 8] [--temperature 0.2]
Writes reports/prompt_comparison.{md,json}.
"""
from __future__ import annotations

import argparse

from _eval_common import STRATEGIES, load_sample, make_client, markdown_table, mock_banner, score_one, summarize, write_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--n-per-label", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    client = make_client(args.provider, args.model)
    sample = load_sample(args.n_per_label, args.seed)
    print(f"Provider {client.provider_name}/{client.model}; {len(sample)} answers x {len(STRATEGIES)} strategies")
    rows_by, summaries = {}, {}
    for strat in STRATEGIES:
        rows = [score_one(client, strat, r, args.temperature) for r in sample.itertuples()]
        rows_by[strat], summaries[strat] = rows, summarize(rows)
        print(f"  {strat:18s} {summaries[strat]}")

    cols = ["valid_rate", "first_try_valid_rate", "score_mean", "score_sd", "spearman_vs_label", "band_accuracy", "mean_latency_ms"]
    body = (mock_banner(client.provider_name)
            + f"Provider: `{client.provider_name}` / `{client.model}` · temperature {args.temperature} · "
              f"{len(sample)} stratified answers (seed {args.seed}).\n\n"
            + "**Strategies**\n\n" + "\n".join(f"- `{k}`: {v}" for k, v in STRATEGIES.items()) + "\n\n"
            + markdown_table(summaries, cols)
            + "\n**How to read this.** `valid_rate` = output parsed and matched the Pydantic schema within the retry "
              "budget; `first_try_valid_rate` = valid without needing a corrective retry; `spearman_vs_label` = rank "
              "correlation between the 0-100 rubric score and the poor/average/good label; `band_accuracy` = score "
              "snapped to the nearest label anchor.\n\n"
              "**Caveats.** Labels are LLM-authored and unreviewed; the sample is tiny, so differences of a few "
              "points are noise. Chain-of-thought costs extra tokens/latency, and its reasoning text is not shown "
              "to users.\n")
    print("Report:", write_report("prompt_comparison", "Prompt strategy comparison", body,
                                  {"provider": client.provider_name, "model": client.model, "summaries": summaries,
                                   "rows": rows_by}))


if __name__ == "__main__":
    main()
