"""Offline wiring test for the LoRA and DistilBERT scripts using tiny RANDOM models (no downloads).

It proves the code path (split -> LoRA wrap -> train -> save -> reload -> predict) works and that the reloaded
adapter reproduces the in-memory model's predictions. It says nothing about real model quality.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import finetune_lora as F  # noqa: E402


def test_split_is_grouped_by_question():
    tr, te = F.load_split()
    assert set(tr.question_id).isdisjoint(set(te.question_id))
    assert len(tr) > len(te) > 0


def test_lora_train_save_reload_roundtrip(tmp_path):
    tr, te = F.load_split()
    texts = [F.to_prompt(q, a) for q, a in zip(tr.question, tr.answer)] + tr.feedback.tolist()
    model, tok = F.load_base("unused", True, texts)
    F.save_random_base(model, tmp_path)
    model = F.add_lora(model, r=4)
    summ = F.trainable_summary(model)
    assert 0 < summ["percent"] < 10  # only the adapters are trainable
    losses = F.train(model, tok, tr.head(48), epochs=3, batch_size=8, lr=3e-3, log=lambda *_: None)
    assert losses[-1] < losses[0]
    sub = te.head(6)
    live = F.generate_feedback(model, tok, sub.question.tolist(), sub.answer.tolist(), max_new_tokens=20)
    F.save(model, tok, tmp_path, {"base": "unused", "smoke": True})
    m2, tok2, _ = F.load_for_inference(tmp_path)
    assert F.generate_feedback(m2, tok2, sub.question.tolist(), sub.answer.tolist(), max_new_tokens=20) == live


def test_evaluate_reports_groundedness_metrics():
    tr, te = F.load_split()
    texts = [F.to_prompt(q, a) for q, a in zip(tr.question, tr.answer)] + tr.feedback.tolist()
    model, tok = F.load_base("unused", True, texts)
    res = F.evaluate(model.eval(), tok, te.head(4))
    assert {"format_ok", "verdict_correct", "evidence_star_ok", "improve_is_weakest", "rouge_l"} <= set(res)
