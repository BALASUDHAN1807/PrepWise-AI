"""Discriminative answer-quality classifier (Unit 1: generative vs discriminative AI).

The *generative* LLM writes rubric scores and feedback; this *discriminative* model only
maps an answer to one of three labels (poor / average / good).

* Default model: TF-IDF (word 1-2 grams) + Logistic Regression, trained on
  ``data/soft_skill_labeled.csv`` and cached in ``models_store/``.
* Optional: a fine-tuned DistilBERT (see ``scripts/train_distilbert_classifier.py``). If the
  directory ``models_store/distilbert_softskill`` exists and ``transformers`` can load it, its
  prediction is returned alongside the TF-IDF one.

Honest evaluation: :func:`evaluate` reports a random stratified hold-out AND a
leave-questions-out (grouped) cross-validation, which is the realistic estimate for unseen
questions. The dataset is small and LLM-authored, so treat numbers as indicative only.
"""
from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from backend.core.config import get_settings

log = logging.getLogger("prepwise.classifier")
LABELS = ["poor", "average", "good"]
MODEL_FILE = "softskill_tfidf_lr.joblib"


def dataset_path() -> Path:
    return get_settings().data_path / "soft_skill_labeled.csv"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(dataset_path())
    df = df.dropna(subset=["answer", "label"])
    return df[df["label"].isin(LABELS)].reset_index(drop=True)


def structural_features(texts) -> np.ndarray:
    """Hand-crafted surface cues a recruiter also looks at: length, numbers, STAR cues, hedges, fillers.

    Module-level (not a lambda) so the fitted pipeline can be pickled with joblib.
    """
    from backend.services import heuristics as H

    rows = []
    for t in texts:
        wc = len(H.words(t))
        star = H.detect_star(t)
        low = t.lower()
        fillers = sum(H.count_fillers(t).values())
        hedges = sum(low.count(h) for h in H.HEDGES)
        rows.append([math.log1p(wc), min(wc, 120) / 120, float(bool(re.search(r"\d", t))),
                     len(re.findall(r"\d+", t)), sum([star.situation, star.task, star.action, star.result]),
                     float(star.result), hedges, 100 * fillers / max(1, wc),
                     float(bool(re.search(r"learn|next time|in hindsight|realis|realiz", low)))])
    return np.asarray(rows, dtype=float)


def build_pipeline(variant: str = "tfidf+features") -> Pipeline:
    """``tfidf_only`` is the plain bag-of-words baseline; ``tfidf+features`` adds structural cues."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    if variant == "tfidf_only":
        feats = tfidf
    else:
        feats = FeatureUnion([
            ("tfidf", tfidf),
            ("struct", Pipeline([("f", FunctionTransformer(structural_features)), ("scale", StandardScaler())])),
        ])
    return Pipeline([("feats", feats), ("lr", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))])


def train(save: bool = True) -> Pipeline:
    df = load_dataset()
    pipe = build_pipeline().fit(df["answer"].tolist(), df["label"])
    if save:
        out = get_settings().models_path
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out / MODEL_FILE)
    get_model.cache_clear()
    return pipe


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    path = get_settings().models_path / MODEL_FILE
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:  # incompatible sklearn version etc.
            log.warning("could not load cached classifier (%s); retraining", type(exc).__name__)
    return train(save=True)


def predict(answer: str) -> dict:
    """Label + class probabilities (TF-IDF/LR), plus DistilBERT output when available."""
    pipe = get_model()
    proba = pipe.predict_proba([answer])[0]
    classes = list(pipe.classes_)
    probs = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    label = max(probs, key=probs.get)
    out = {"label": label, "confidence": probs[label], "probabilities": probs,
           "model": "tfidf+logistic-regression", "distilbert": distilbert_predict(answer)}
    return out


@lru_cache(maxsize=1)
def _distilbert():
    d = get_settings().models_path / "distilbert_softskill"
    if not d.exists():
        return None
    try:
        from transformers import pipeline

        return pipeline("text-classification", model=str(d), top_k=None)
    except Exception as exc:
        log.warning("DistilBERT classifier unavailable (%s)", type(exc).__name__)
        return None


def distilbert_predict(answer: str) -> dict | None:
    pipe = _distilbert()
    if pipe is None:
        return None
    try:
        scores = pipe(answer[:2000])
        scores = scores[0] if scores and isinstance(scores[0], list) else scores
        probs = {s["label"]: round(float(s["score"]), 4) for s in scores}
        label = max(probs, key=probs.get)
        return {"label": label, "confidence": probs[label], "probabilities": probs, "model": "distilbert-finetuned"}
    except Exception as exc:  # pragma: no cover
        log.warning("DistilBERT inference failed: %s", exc)
        return None


def _metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4), "precision_macro": round(p, 4),
            "recall_macro": round(r, 4), "f1_macro": round(f, 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(), "labels": LABELS}


def _cv_summary(make_model, X, y, seed: int) -> dict:
    """Repeated stratified 5-fold CV (3 repeats): mean and std over folds, far less noisy than one split."""
    accs, f1s = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed).split(X, y):
        m = make_model().fit([X[i] for i in tr], y[tr])
        pred = m.predict([X[i] for i in te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(precision_recall_fscore_support(y[te], pred, labels=LABELS, average="macro", zero_division=0)[2])
    return {"accuracy_mean": round(float(np.mean(accs)), 4), "accuracy_std": round(float(np.std(accs)), 4),
            "f1_macro_mean": round(float(np.mean(f1s)), 4), "folds": len(accs)}


class _LengthOnly:
    """Baseline that sees only the word count."""

    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)

    @staticmethod
    def _x(texts):
        return np.array([[len(t.split())] for t in texts], dtype=float)

    def fit(self, texts, y):
        self.lr.fit(self._x(texts), y)
        return self

    def predict(self, texts):
        return self.lr.predict(self._x(texts))


def evaluate(seed: int = 42) -> dict:
    """Compare model variants under two protocols.

    * ``random_cv``: repeated stratified 5-fold. Optimistic, because answers to the same question
      share vocabulary and sentence fragments across folds.
    * ``leave_questions_out``: GroupKFold by question. The realistic estimate for new questions.

    ``selected`` (the shipped model) also gets a stratified hold-out confusion matrix for the
    notebook; with only ~45 test answers that single number is noisy, so prefer the CV figures.
    """
    df = load_dataset()
    X, y = df["answer"].tolist(), df["label"].to_numpy()
    groups = df["question_id"] if "question_id" in df else pd.Series(range(len(df)))
    makers = {"length_only": _LengthOnly, "tfidf_only": lambda: build_pipeline("tfidf_only"),
              "tfidf+features": lambda: build_pipeline("tfidf+features")}
    variants = {}
    for name, make in makers.items():
        preds = np.empty(len(df), dtype=object)
        for tr, te in GroupKFold(n_splits=4).split(df, groups=groups):
            preds[te] = make().fit([X[i] for i in tr], y[tr]).predict([X[i] for i in te])
        variants[name] = {"random_cv": _cv_summary(make, X, y, seed), "leave_questions_out": _metrics(y, list(preds))}
    tr_idx, te_idx = train_test_split(range(len(df)), test_size=0.25, stratify=y, random_state=seed)
    hold = build_pipeline("tfidf+features").fit([X[i] for i in tr_idx], y[tr_idx])
    return {"n": len(df), "selected": "tfidf+features", "variants": variants,
            "holdout_selected": _metrics(y[te_idx], hold.predict([X[i] for i in te_idx])),
            "note": "Small LLM-authored dataset in which length and structure strongly track the label "
                    "(see the length_only baseline). Accuracy here does NOT show real-world interview-grading skill."}
