"""OPTIONAL: fine-tune DistilBERT as the discriminative model (Unit 4) and compare it with TF-IDF + LR.

The backend loads it automatically when ``<MODELS_DIR>/distilbert_softskill`` exists and shows it next to the
classical classifier. Needs `torch` + `transformers`; a GPU is helpful but not required (204 short answers).

Usage
    python scripts/train_distilbert_classifier.py --epochs 4
    python scripts/train_distilbert_classifier.py --smoke-test --out /tmp/distilbert_smoke   # offline wiring check only
Evaluation uses a question-grouped hold-out (unseen questions), the same split as scripts/finetune_lora.py.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from finetune_lora import LABELS, load_split  # noqa: E402  (labels + the shared question-grouped split)
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from backend.core.config import get_settings  # noqa: E402


def tiny_bert_for_smoke_test(texts: list[str]):
    """Randomly initialised ~0.5M-parameter BERT classifier + locally trained tokenizer (offline wiring test)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1500, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                   sep_token="[SEP]", model_input_names=["input_ids", "attention_mask"])
    cfg = BertConfig(vocab_size=len(fast), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=128, max_position_embeddings=512, num_labels=3,
                     id2label=dict(enumerate(LABELS)), label2id={lab: i for i, lab in enumerate(LABELS)})
    return BertForSequenceClassification(cfg), fast


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="distilbert-base-uncased")
    ap.add_argument("--out", default=str(get_settings().models_path / "distilbert_softskill"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tr, te = load_split(seed=args.seed)
    if args.smoke_test:
        model, tok = tiny_bert_for_smoke_test(tr["answer"].tolist() + te["answer"].tolist())
        args.lr = max(args.lr, 3e-3)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={lab: i for i, lab in enumerate(LABELS)})
        tok = AutoTokenizer.from_pretrained(args.base)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    X, y = tr["answer"].tolist(), [LABELS.index(label) for label in tr["label"]]
    idx = list(range(len(X)))
    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(idx)
        total = 0.0
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            enc = tok([X[j] for j in b], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            enc.pop("token_type_ids", None)  # DistilBERT has no segment embeddings
            loss = model(**enc, labels=torch.tensor([y[j] for j in b])).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item() * len(b)
        print(f"epoch {ep}/{args.epochs}  loss {total / len(X):.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 16):
            enc = tok(te["answer"].tolist()[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            enc.pop("token_type_ids", None)  # DistilBERT has no segment embeddings
            preds += [LABELS[k] for k in model(**enc).logits.argmax(-1).tolist()]
    acc, f1 = accuracy_score(te["label"], preds), f1_score(te["label"], preds, labels=LABELS, average="macro")
    print(f"unseen-question hold-out: accuracy {acc:.3f}, macro-F1 {f1:.3f} (n={len(te)}, {te.question_id.nunique()} questions)")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved ->", args.out)
    if args.smoke_test:
        print("SMOKE TEST ONLY: random tiny model; numbers are meaningless.")


if __name__ == "__main__":
    main()
