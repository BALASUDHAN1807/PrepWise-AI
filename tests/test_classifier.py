"""Discriminative answer classifier (TF-IDF + Logistic Regression)."""
import pandas as pd
import pytest

from backend.services import classifier_service as C

GOOD = ("In my final-year project two teammates disagreed on which database to use, and the argument blocked us for a "
        "week. As team lead I set up a small benchmark of both options, we reviewed the numbers together, chose MySQL "
        "and delivered four days early with 30 percent fewer query bugs. I learned that moving a debate to data keeps "
        "it about the problem and not the people.")
# NOTE: a concise ~40-word strong answer is rated 'average' because every 'good' training answer is 53+ words.
# That known limitation is documented in data/DATASET_CARD.md and docs/responsible_ai.md.
POOR = "Um, I don't know, I think maybe it was fine, basically."


def test_dataset_meets_requirements():
    df = C.load_dataset()
    assert len(df) >= 150 and set(df["label"]) == set(C.LABELS)
    assert df["label"].value_counts().min() >= 50
    assert df["answer"].is_unique


def test_dataset_is_flagged_as_unreviewed():
    raw = pd.read_csv(C.dataset_path())
    assert "reviewed" in raw and set(raw["reviewed"]) <= {"yes", "no"}


def test_train_predict_orders_quality():
    C.train(save=True)
    good, poor = C.predict(GOOD), C.predict(POOR)
    assert good["label"] == "good" and poor["label"] == "poor"
    assert good["probabilities"]["good"] > good["probabilities"]["poor"]
    assert pytest.approx(sum(good["probabilities"].values()), abs=1e-3) == 1.0
    assert good["distilbert"] is None  # optional model not installed in tests


def test_evaluate_reports_all_metrics_and_baselines():
    ev = C.evaluate()
    assert ev["n"] >= 150 and ev["selected"] == "tfidf+features"
    for name, v in ev["variants"].items():
        m = v["leave_questions_out"]
        assert {"accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix"} <= set(m), name
        assert len(m["confusion_matrix"]) == 3 and {"accuracy_mean", "accuracy_std"} <= set(v["random_cv"])
    assert {"accuracy", "confusion_matrix"} <= set(ev["holdout_selected"])


def test_selected_model_clearly_beats_length_only_baseline():
    """Guards against the dataset collapsing back to 'longer answer = better answer'."""
    v = C.evaluate()["variants"]
    sel, base, bow = v["tfidf+features"], v["length_only"], v["tfidf_only"]
    assert sel["leave_questions_out"]["accuracy"] >= 0.80          # unseen questions
    assert sel["leave_questions_out"]["accuracy"] >= base["leave_questions_out"]["accuracy"] + 0.10
    assert sel["leave_questions_out"]["accuracy"] >= bow["leave_questions_out"]["accuracy"]


def test_padded_distractors_break_the_length_shortcut():
    df = C.load_dataset()
    wc = df["answer"].str.split().str.len()
    long_weak = df[(wc >= 53) & (df["label"] != "good")]
    assert len(long_weak) >= 10
