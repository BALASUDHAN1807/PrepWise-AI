"""LoRA fine-tuning of a small seq2seq model (Flan-T5) for interview-FEEDBACK GENERATION - Unit 3/4 (adaptation).

Task (text-to-text)
    input : "Write feedback for this interview answer.\\nQuestion: ...\\nAnswer: ..."
    output: "Verdict: average. Evidence: 52 words; STAR parts found: action, result; includes measurable detail.
             Strength: ... Improve depth: ... Next step: ..."

The training data is data/feedback_sft.jsonl, a SYNTHETIC set composed from labels + a rule-based rubric
(see scripts/build_feedback_dataset.py and data/DATASET_CARD.md). Rows carry ``reviewed: false`` until a human checks
them; use ``--only-reviewed`` to train on reviewed rows only.

Why LoRA? Well under 2% of parameters (low-rank adapters on the attention q/v projections) are trained, so a laptop CPU or a
free Colab GPU is enough and the adapter is a few MB. The base weights stay frozen.

Evaluation protocol: the split is by QUESTION (test = unseen questions) and generated text is audited against
measurable facts (verdict, STAR parts, numbers, weakest criterion) - see scripts/_feedback_eval.py. Because the
reference text is templated, a good score shows the model learned the format and the grounding rules, NOT that its
advice is pedagogically better than an LLM's.

Usage
    python scripts/finetune_lora.py --base google/flan-t5-small --epochs 10 --out models/lora_flan_t5_small
    python scripts/finetune_lora.py --smoke-test      # offline wiring check with a tiny random T5 (NOT a real model)

The same functions are imported by notebooks/finetune_lora.ipynb and scripts/compare_approaches.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402
from _feedback_eval import aggregate, score_feedback  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

PROMPT = "Write feedback for this interview answer.\nQuestion: {q}\nAnswer: {a}"
MAX_TARGET = 160


# ------------------------------------------------------------------ data ----
def load_feedback_data(only_reviewed: bool = False) -> pd.DataFrame:
    df = pd.read_json(ROOT / "data" / "feedback_sft.jsonl", lines=True)
    return df[df["reviewed"]].reset_index(drop=True) if only_reviewed else df


def load_split(test_frac: float = 0.25, seed: int = 42, only_reviewed: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Question-grouped split (deterministic): every answer to a question lands on the same side."""
    df = load_feedback_data(only_reviewed)
    if df.empty:
        raise SystemExit("No rows to train on (did you pass --only-reviewed with nothing marked reviewed?).")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed).split(df, groups=df["question_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def to_prompt(question: str, answer: str) -> str:
    return PROMPT.format(q=question, a=answer)


# ----------------------------------------------------------------- models ----
def tiny_t5_for_smoke_test(texts: list[str]):
    """A ~0.3M-parameter randomly initialised T5 + a tokenizer trained on the data. Offline wiring test only."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1500, special_tokens=["<pad>", "</s>", "<unk>"]))
    tok.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", tok.token_to_id("</s>"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    cfg = T5Config(vocab_size=len(fast), d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    return T5ForConditionalGeneration(cfg), fast


def load_base(base: str, smoke: bool, texts: list[str] | None = None):
    if smoke:
        return tiny_t5_for_smoke_test(texts or [])
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    return AutoModelForSeq2SeqLM.from_pretrained(base), AutoTokenizer.from_pretrained(base)


def add_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05, target_modules=("q", "v")):
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=list(target_modules))
    return get_peft_model(model, cfg)


def trainable_summary(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "percent": round(100 * trainable / total, 3)}


# ---------------------------------------------------------------- training ---
def train(model, tok, train_df: pd.DataFrame, epochs: int = 10, batch_size: int = 8, lr: float = 3e-4, seed: int = 42,
          max_len: int = 384, log=print) -> list[float]:
    """Plain PyTorch loop (no Trainer) so every step is visible. Returns mean loss per epoch."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    src = [to_prompt(q, a) for q, a in zip(train_df["question"], train_df["answer"])]
    tgt = train_df["feedback"].tolist()
    idx = list(range(len(src)))
    losses = []
    model.train()
    for ep in range(1, epochs + 1):
        random.shuffle(idx)
        total, n = 0.0, 0
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            enc = tok([src[j] for j in b], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            lab = tok([tgt[j] for j in b], return_tensors="pt", padding=True, truncation=True, max_length=MAX_TARGET).input_ids
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item() * len(b)
            n += len(b)
        losses.append(total / n)
        log(f"epoch {ep}/{epochs}  loss {losses[-1]:.4f}")
    return losses


def generate_feedback(model, tok, questions: list[str], answers: list[str], batch_size: int = 8, max_len: int = 384,
                      max_new_tokens: int = 110) -> list[str]:
    """Greedy decoding (deterministic)."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(answers), batch_size):
            prompts = [to_prompt(q, a) for q, a in zip(questions[i:i + batch_size], answers[i:i + batch_size])]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            out += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return out


def evaluate(model, tok, test_df: pd.DataFrame, keep_samples: int = 3) -> dict:
    """Generate for every test answer and audit the text (format, verdict, evidence, weakest criterion, strength, ROUGE-L)."""
    t0 = time.perf_counter()
    preds = generate_feedback(model, tok, test_df["question"].tolist(), test_df["answer"].tolist())
    ms = (time.perf_counter() - t0) * 1000 / len(test_df)
    rows = [score_feedback(p, r.question, r.answer, r.label, r.feedback) for p, r in zip(preds, test_df.itertuples())]
    return {**aggregate(rows), "ms_per_answer": round(ms, 1), "predictions": preds,
            "samples": [{"label": r.label, "generated": p, "reference": r.feedback}
                        for p, r in list(zip(preds, test_df.itertuples()))[:keep_samples]]}


def save(model, tok, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)  # for a PEFT model this writes only the adapter
    tok.save_pretrained(out)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def save_random_base(model, out: Path) -> None:
    """Smoke test only: its "base" model is randomly initialised, so it must be stored to be reloadable.

    Must be called BEFORE :func:`add_lora` - afterwards the layers are wrapped and the keys no longer match.
    """
    model.save_pretrained(out / "base_model")


def load_for_inference(adapter_dir: str | Path):
    """Load base + LoRA adapter saved by :func:`save` (works for real runs and the smoke test)."""
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    adapter_dir = Path(adapter_dir)
    meta = json.loads((adapter_dir / "run_meta.json").read_text())
    base_path = str(adapter_dir / "base_model") if meta.get("smoke") else meta["base"]
    base = AutoModelForSeq2SeqLM.from_pretrained(base_path)
    return PeftModel.from_pretrained(base, adapter_dir), AutoTokenizer.from_pretrained(adapter_dir), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="google/flan-t5-small")
    ap.add_argument("--out", default=str(ROOT / "models" / "lora_flan_t5_small"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-reviewed", action="store_true", help="train/test only on rows with reviewed=true")
    ap.add_argument("--smoke-test", action="store_true", help="tiny random T5 + local tokenizer; verifies wiring offline")
    args = ap.parse_args()

    tr, te = load_split(seed=args.seed, only_reviewed=args.only_reviewed)
    both = pd.concat([tr, te])
    texts = [to_prompt(q, a) for q, a in zip(both["question"], both["answer"])] + both["feedback"].tolist()
    model, tok = load_base(args.base, args.smoke_test, texts)
    base_params = sum(p.numel() for p in model.parameters())
    if args.smoke_test:
        save_random_base(model, Path(args.out))
    model = add_lora(model, r=args.rank)
    summ = trainable_summary(model)
    print(f"base params {base_params:,} · LoRA trainable {summ['trainable']:,} ({summ['percent']}%)")
    print(f"train {len(tr)} answers / {tr.question_id.nunique()} questions · test {len(te)} answers / "
          f"{te.question_id.nunique()} UNSEEN questions · reviewed rows in train: {int(tr.reviewed.sum())}")
    lr = max(args.lr, 3e-3) if args.smoke_test else args.lr
    losses = train(model, tok, tr, args.epochs, args.batch_size, lr, args.seed)
    res = evaluate(model, tok, te)
    print({k: v for k, v in res.items() if k not in ("predictions", "samples")})
    for s in res["samples"]:
        print(f"\n[{s['label']}] generated: {s['generated']}\n        reference: {s['reference']}")
    meta = {"base": args.base, "smoke": args.smoke_test, "epochs": args.epochs, "lr": lr, "rank": args.rank,
            "seed": args.seed, "losses": losses, "trainable": summ, "only_reviewed": args.only_reviewed,
            "test_metrics": {k: v for k, v in res.items() if k not in ("predictions", "samples")},
            "split": "GroupShuffleSplit by question_id, test_frac=0.25",
            "dataset": "data/feedback_sft.jsonl (synthetic, template-composed; see data/DATASET_CARD.md)"}
    save(model, tok, Path(args.out), meta)
    print("\nsaved adapter ->", args.out)
    if args.smoke_test:
        print("SMOKE TEST ONLY: random tiny model - metrics are meaningless; this proves the training/saving/loading code runs.")


if __name__ == "__main__":
    main()
