"""Prompting vs RAG vs LoRA fine-tuning for FEEDBACK GENERATION - Unit 3/4 synthesis.

All approaches write feedback for the SAME held-out answers (question-grouped split -> unseen questions) in the same
four-part format, and every output is audited against measurable facts (scripts/_feedback_eval.py):

    nn_copy      : no model - copies the feedback of the most similar TRAINING answer (retrieval-only baseline)
    prompting    : zero-shot LLM prompt with format instructions, no examples
    rag_fewshot  : same prompt + the k most similar training answers WITH their feedback (retrieval-augmented)
    lora         : fine-tuned Flan-T5 adapter from scripts/finetune_lora.py (skipped if not trained)

Usage:  python scripts/compare_approaches.py [--provider gemini] [--adapter models/lora_flan_t5_small] [--k 3] [--limit 20]
Writes reports/compare_approaches.{md,json}.

How to read it: the reference text is templated, so ROUGE-L rewards models that learned the template. The groundedness
columns (verdict, evidence, weakest-criterion, strength) say whether claims are TRUE of the answer according to the
rule-based rubric. Neither measures teaching quality - that needs human raters (see docs/responsible_ai.md).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from _eval_common import ROOT, make_client, markdown_table, mock_banner, write_report
from _feedback_eval import aggregate, score_feedback
from finetune_lora import generate_feedback as lora_generate
from finetune_lora import load_for_inference, load_split
from pydantic import BaseModel

from backend.llm.client import LLMError
from backend.services import rag_service as R

SYSTEM = (
    "You write feedback for campus-placement interview answers. Reply as JSON {\"feedback\": \"...\"} where the feedback "
    "text follows EXACTLY this format on one line:\n"
    "Verdict: <poor|average|good>. Evidence: <N> words; STAR parts found: <comma-separated subset of situation, task, "
    "action, result, or none>; <includes|no> measurable detail. Strength: <one sentence, or 'No single strength stands "
    "out yet; the attempt to answer is a start.'> Improve <weakest of clarity|structure|relevance|confidence|"
    "grammar_vocabulary|depth>: <one imperative sentence>. Next step: <one imperative sentence>.\n"
    "Poor = vague/off-topic/one-liner; average = relevant but generic or missing a result; good = specific, structured, "
    "clear outcome. Judge content only; never mention name, gender, college or region. Claim only what is true of the answer."
)


class FeedbackOut(BaseModel):
    feedback: str


def llm_feedback(client, question: str, answer: str, shots: list[tuple[str, str, str]] | None) -> str:
    ex = ""
    if shots:
        ex = "Graded examples of similar answers (same format):\n" + "\n".join(
            f'- Question: {q}\n  Answer: "{a[:350]}"\n  Feedback: {fb}' for q, a, fb in shots) + "\n\n"
    user = f"{ex}Question: {question}\nAnswer: {answer}"
    try:
        res = client.complete_json(SYSTEM, user, FeedbackOut, temperature=0.2,
                                   task="exp_feedback_rag" if shots else "exp_feedback_zero",
                                   variables={"question": question, "answer": answer})
        return res.data.feedback
    except (LLMError, ValueError):
        return ""


def run(name, fn, test, results, outputs, extra=None):
    t0 = time.perf_counter()
    preds = fn()
    ms = (time.perf_counter() - t0) * 1000 / len(test)
    rows = [score_feedback(p, r.question, r.answer, r.label, r.feedback) for p, r in zip(preds, test.itertuples())]
    results[name] = {**aggregate(rows), "ms_per_answer": round(ms, 1)}
    outputs[name] = preds[:2]
    print(f"{name:12s} {results[name]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--adapter", default=str(ROOT / "models" / "lora_flan_t5_small"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test answers (cost control)")
    args = ap.parse_args()

    train, test = load_split(seed=args.seed)
    if args.limit:
        test = test.head(args.limit)
    client = make_client(args.provider, args.model)
    emb = R.embed(train["answer"].tolist())
    sims = R.cosine_matrix(R.embed(test["answer"].tolist()), emb)  # retrieval index = TRAINING answers only
    nearest = [list(row.argsort()[::-1][:args.k]) for row in sims]

    results, outputs, notes = {}, {}, []
    run("nn_copy", lambda: [train.feedback[idx[0]] for idx in nearest], test, results, outputs)
    run("prompting", lambda: [llm_feedback(client, r.question, r.answer, None) for r in test.itertuples()], test, results, outputs)
    run("rag_fewshot", lambda: [llm_feedback(client, r.question, r.answer,
                                            [(train.question[j], train.answer[j], train.feedback[j]) for j in nearest[i]])
                                for i, r in enumerate(test.itertuples())], test, results, outputs)
    info = R.embedding_info()
    if not info.get("semantic"):
        notes.append(f"Retrieval used `{info['embedder']}` (hashing fallback, not semantic); with all-MiniLM-L6-v2 the "
                     "retrieved examples (and the nn_copy / rag rows) would likely be better.")

    if Path(args.adapter, "run_meta.json").exists():
        model, tok, meta = load_for_inference(args.adapter)
        run("lora", lambda: lora_generate(model, tok, test["question"].tolist(), test["answer"].tolist()), test, results, outputs)
        if meta.get("smoke"):
            notes.append("The `lora` row comes from the offline SMOKE-TEST adapter (tiny random model) - meaningless. Train a real "
                         "one: `python scripts/finetune_lora.py --base google/flan-t5-small`.")
    else:
        notes.append(f"`lora` skipped: no adapter at `{args.adapter}`. Train one with scripts/finetune_lora.py "
                     "(or notebooks/finetune_lora.ipynb on a free Colab GPU).")

    cols = ["n", "format_ok", "verdict_correct", "evidence_star_ok", "evidence_number_ok", "improve_is_weakest",
            "strength_supported", "rouge_l", "ms_per_answer"]
    samples = "\n".join(f"**{k}**\n\n> {v[0] or '(empty / invalid output)'}\n" for k, v in outputs.items())
    body = (mock_banner(client.provider_name)
            + f"LLM: `{client.provider_name}/{client.model}` · train {len(train)} answers ({train.question_id.nunique()} questions) · "
              f"test {len(test)} answers ({test.question_id.nunique()} unseen questions).\n\n"
            + markdown_table(results, cols)
            + ("\n" + "\n".join(f"- {n}" for n in notes) + "\n" if notes else "")
            + "\n### First test answer - what each approach wrote\n\n" + samples
            + "\n**Reading the trade-offs.** Prompting needs no data and can explain in natural language, but costs API calls, "
              "varies run to run and may not follow a strict format. RAG adds in-context examples (no training, easy to update, "
              "longer prompts, quality depends on retrieval). LoRA bakes the format and grounding rules into a small local model "
              "(cheap, private, deterministic) but needs data and compute and can only be as good - and as biased - as its training "
              "text, which here is templated and unreviewed.\n\n"
              "**Caveats.** n is tiny (3 unseen questions): differences of a few points are noise. The reference is templated, "
              "so ROUGE-L favours template learners. Groundedness is judged by the same rule-based rubric that wrote the references.\n")
    print("Report:", write_report("compare_approaches", "Prompting vs RAG vs fine-tuning (feedback generation)", body,
                                  {"results": results, "outputs": outputs, "notes": notes, "provider": client.provider_name}))


if __name__ == "__main__":
    main()
