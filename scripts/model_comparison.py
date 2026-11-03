"""Compare LLM providers/models on the same rubric-scoring task - Unit 4 (model families, trade-offs).

Usage:  python scripts/model_comparison.py --models "gemini,openai,anthropic,ollama:llama3.2"
Models without an API key (or unreachable Ollama) are skipped and listed as such. The rule-based mock
heuristic is always included as a no-LLM baseline. Writes reports/model_comparison.{md,json}.
"""
from __future__ import annotations

import argparse

from _eval_common import get_settings, load_sample, make_client, markdown_table, score_one, summarize, write_report


def available(provider: str) -> tuple[bool, str]:
    s = get_settings()
    keys = {"gemini": s.gemini_api_key, "openai": s.openai_api_key, "anthropic": s.anthropic_api_key}
    if provider in keys and not keys[provider]:
        return False, f"{provider.upper()}_API_KEY not set"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="gemini,openai,anthropic,ollama", help="comma list of provider[:model]")
    ap.add_argument("--strategy", default="structured_output")
    ap.add_argument("--n-per-label", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    sample = load_sample(args.n_per_label)
    targets = [m.strip() for m in args.models.split(",") if m.strip()] + ["mock"]
    summaries, skipped, rows_by = {}, {}, {}
    for t in targets:
        provider, _, model = t.partition(":")
        ok, why = available(provider)
        if not ok:
            skipped[t] = why
            continue
        client = make_client(provider, model or None)
        rows = [score_one(client, args.strategy, r, args.temperature) for r in sample.itertuples()]
        if provider != "mock" and not any(r["valid"] for r in rows):  # e.g. Ollama not running
            skipped[t] = rows[0]["error"] or "no valid output"
            continue
        label = f"{client.provider_name}/{client.model}" + (" (rule-based baseline)" if provider == "mock" else "")
        rows_by[label], summaries[label] = rows, summarize(rows)
        print(f"{label:55s} {summaries[label]}")

    cols = ["valid_rate", "first_try_valid_rate", "score_mean", "score_sd", "mean_latency_ms", "spearman_vs_label", "band_accuracy"]
    body = (f"Strategy `{args.strategy}`, temperature {args.temperature}, {len(sample)} stratified answers.\n\n"
            + markdown_table(summaries, cols)
            + ("\n**Skipped:** " + "; ".join(f"`{k}` ({v})" for k, v in skipped.items()) + "\n" if skipped else "")
            + "\n**Trade-offs to discuss** (fill in from your own run): quality vs latency vs cost; hosted APIs send "
              "answer text to a third party (privacy), while Ollama keeps data local but needs hardware and is "
              "usually weaker at strict JSON; larger models follow the schema more reliably (see first_try_valid_rate).\n\n"
              "**Columns.** score_mean / score_sd = mean and standard deviation of the 0-100 rubric score over the sample; latency is "
              "per answer including retries; `spearman_vs_label` and `band_accuracy` = agreement with the poor/average/good labels "
              "(NOT with independent human scores - none exist yet).\n\n"
              "**Caveats.** Labels are LLM-authored and unreviewed; n is small; the rule-based baseline shows how much "
              "of the label signal is recoverable from surface features alone (length/structure).\n")
    print("Report:", write_report("model_comparison", "LLM provider/model comparison", body,
                                  {"summaries": summaries, "skipped": skipped, "rows": rows_by}))


if __name__ == "__main__":
    main()
