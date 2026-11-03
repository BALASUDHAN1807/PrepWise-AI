# Model cards

Two trained artefacts ship with (or can be produced by) this repo. Neither is validated for real-world assessment.

---
## A. Fine-tuned feedback generator - Flan-T5-small + LoRA (`scripts/finetune_lora.py`)

| | |
|---|---|
| **Intended use** | Course demonstration of parameter-efficient fine-tuning: generate a short, structured feedback paragraph for a practice interview answer |
| **Not intended for** | Grading, ranking or screening real candidates; any high-stakes decision; languages other than English |
| **Base model** | `google/flan-t5-small` (Apache-2.0; ~77M parameters) - or any seq2seq model via `--base` |
| **Adaptation** | LoRA r=8, alpha=16, dropout 0.05 on attention `q` and `v`; only the adapters train (well under 2% of parameters); AdamW, lr 3e-4, 10 epochs, batch 8, seed 42 |
| **Training data** | `data/feedback_sft.jsonl` (synthetic, template-composed, **unreviewed**) - see `data/DATASET_CARD.md` |
| **Split / evaluation** | Split by question (3 unseen test questions, 51 answers). Generated text is audited for format, verdict, evidence, weakest criterion, strength support and ROUGE-L (`scripts/_feedback_eval.py`) |
| **Results** | **Not reported in the repo until you run it** (needs the base weights from Hugging Face). The offline `--smoke-test` uses a tiny *random* model and only checks that the code runs. After a real run, paste `reports/compare_approaches.md` here. |
| **Baselines to compare** | zero-shot prompting, RAG few-shot prompting, nearest-neighbour copy (`scripts/compare_approaches.py`) |
| **Limitations** | Templated targets mean the model may only learn the template; tiny test set; judged by the same rule-based rubric that wrote the targets; hallucination possible for evidence counts (words, STAR parts); English only |
| **Bias and fairness** | Inherits any bias of the dataset recipe and the heuristic rubric (length, formality). No subgroup evaluation has been done |
| **Ethical considerations** | Show output as practice feedback with a "guidance only" disclaimer; keep a human in the loop; obtain consent before using student answers to train |
| **How to reproduce** | `python scripts/build_feedback_dataset.py && python scripts/finetune_lora.py && python scripts/compare_approaches.py` (or `notebooks/finetune_lora.ipynb` on Colab) |
| **Artefacts** | Adapter folder `models/lora_flan_t5_small/` (adapter weights + tokenizer + `run_meta.json`); not committed |
| **If you publish it** | Upload the adapter to the Hugging Face Hub with this card as `README.md` and state the base model licence |

---
## B. Answer-quality classifier - TF-IDF + structural features + Logistic Regression (`backend/services/classifier_service.py`)

| | |
|---|---|
| **Intended use** | Second opinion shown beside the LLM rubric: labels an answer poor / average / good with class probabilities |
| **Not intended for** | Real candidate evaluation; non-English text |
| **Model** | `TfidfVectorizer(1-2 grams, min_df=2, sublinear_tf)` + 9 structural features (length, numbers, STAR cues, hedges, fillers, reflection) -> `LogisticRegression(C=2, class_weight=balanced)` |
| **Training data** | `data/soft_skill_labeled.csv` (204 synthetic, unreviewed answers; 12 questions) |
| **Evaluation** (run `python scripts/train_classifier.py`) | Leave-questions-out accuracy / macro-F1: length-only baseline 0.68 / 0.67; TF-IDF only 0.78 / 0.78; **TF-IDF + features 0.89 / 0.89**; random 5-fold CV (optimistic): 0.91 +/- 0.05. Hold-out confusion matrix is in `reports/classifier_eval.md` |
| **How to read these numbers** | They measure agreement with *synthetic labels* on a small set. Length alone is a strong predictor, so high accuracy does not imply understanding of answer quality |
| **Known failure modes** | Short but excellent answers rated "average"; verbose empty answers can score higher than they should; unseen phrasing/dialects; non-English |
| **Fairness check** | Counterfactual identity-sentence probe in `notebooks/classifier_training.ipynb` (smoke test only, not an audit) |
| **Optional upgrade** | DistilBERT fine-tune (`scripts/train_distilbert_classifier.py`); loaded automatically from `<MODELS_DIR>/distilbert_softskill` if present and displayed beside the classical model. Not trained in this repo - its numbers must come from your own run |
| **Maintenance** | Retrain when the dataset changes; the app trains and caches a model on first use if none exists |
