# Generative vs discriminative AI in PrepWise AI (Unit 1)

*Syllabus link: foundations of generative AI, generative vs discriminative models, VAEs/GANs as background.*

## The distinction in one paragraph
A **discriminative** model learns `P(label | input)` - it draws a boundary and answers "which class is this?".
A **generative** model learns (or is conditioned to produce) `P(output | input)` over rich outputs such as text - it answers
"what would a plausible response look like?". Discriminative models are usually smaller, cheaper, faster and easier to audit;
generative models are flexible but can be confidently wrong, vary between runs, and are harder to test.

## Where each type is used in this project

| Feature | Type | Model | Why this type |
|---|---|---|---|
| Interview question generation (DSA, HR) | Generative | LLM (+ RAG over `question_bank.json`) | Needs fresh, personalised wording; grounded by retrieval and verified by running the reference solution |
| Resume structured extraction | Generative + rules | spaCy + LLM -> Pydantic JSON | Free-form resumes; the JSON schema constrains the output |
| Bullet rewriter, cover letter | Generative | LLM | Open-ended writing |
| Rubric scoring of soft-skill answers (6 criteria + reasons) | Generative | LLM, temperature 0.2 | Needs *explanations* a student can act on |
| **Answer-quality classifier (poor/average/good)** | **Discriminative** | **TF-IDF + structural features + Logistic Regression** (optional DistilBERT) | Cheap, deterministic, inspectable; shown **side by side** with the rubric as a second opinion |
| Embedding-based JD match and course retrieval | Discriminative-style encoder | all-MiniLM-L6-v2 | Maps text to vectors for similarity; no text generation |
| Sentiment / tone | Discriminative | transformer sentiment model (lexicon fallback) | A classification problem |
| Speech-to-text | Sequence model (Whisper) | faster-whisper | Transcription |
| Feedback chain, roadmap, translation | Generative | LLM, fixed prompt chain | Narrative output, validated against schemas and the course catalog |
| Feedback generation (experiment) | Generative, adapted | Flan-T5 + LoRA | Compared with prompting and RAG (`scripts/compare_approaches.py`) |
| **DSA correctness** | **Neither - deterministic** | Sandboxed test execution | A model should never decide whether code is correct |

### Why showing both evaluators side by side is useful
The rubric (generative) gives reasons; the classifier (discriminative) gives a fast, reproducible label with calibrated
probabilities. Where they disagree the UI shows both, which is itself a signal that the answer deserves human attention. The
overall soft-skill score weights them 70% rubric / 20% classifier expectation / 10% delivery analytics (a pure function in
`backend/services/evaluator.py`, unit-tested).

### An honest finding about the discriminative model
On our small synthetic dataset, **answer length alone** gets about 68% accuracy (leave-questions-out), TF-IDF alone about 78%,
and TF-IDF + structural features about 89% (`reports/classifier_eval.md`, regenerate with `python scripts/train_classifier.py`).
Those numbers show the model learned surface cues that also separate *our* synthetic labels; they do **not** show it can grade
real interview answers (see `data/DATASET_CARD.md` and `docs/responsible_ai.md`). Known failure: short but excellent answers
tend to be rated lower.

## Background: VAEs and GANs (not central to this project)
We do **not** train or use VAEs or GANs. For completeness:

* **Variational autoencoder (VAE)** - an encoder maps input `x` to a distribution over a latent code `z`, a decoder reconstructs
  `x`; training maximises a lower bound (reconstruction term + KL regulariser). Good for smooth latent spaces, anomaly detection
  and generating data *similar* to the training set; outputs tend to be blurry for images.
* **Generative adversarial network (GAN)** - a generator tries to fool a discriminator that tries to tell real from fake; the
  two-player game yields sharp samples but training can be unstable (mode collapse). Notably, the discriminator is itself a
  discriminative model - a neat link between the two families.
* **What they could do here (future work)**: privacy-preserving *synthetic tabular data* of student score histories for the admin
  dashboard (e.g. a tabular VAE/CTGAN) so colleges can share benchmarks without sharing real records; synthetic *voice* samples
  to test the speech pipeline across accents; synthetic video for avatar interviewers (diffusion/GAN-family models). For text
  data, today's practical route is LLM-based generation plus **human review** - which is what we document for our datasets.
* **Why not now**: our data is small, text-based and better served by LLMs; adding a GAN/VAE would add complexity without
  improving any feature, and generated data would need the same human review anyway.

## Ethics note (Unit 1)
Generative output can be fluent and wrong. PrepWise mitigates this with structure (Pydantic-validated JSON), grounding (RAG,
catalog validation, running reference solutions), determinism where possible (sandbox for code), visible explanations and
a "scores are guidance only" disclaimer. See `docs/responsible_ai.md`.
