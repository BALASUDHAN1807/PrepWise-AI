# Responsible AI

PrepWise AI scores people. That is a position of influence even when the scores are only "guidance", so this document states what
the system does, what it cannot do, and what you must change before using it with real students. It is written for the
course (Unit 1 ethics; Unit 3 ethical and compliance considerations) and for anyone deploying the code.

## 1. Bias and fairness in scoring
**Risks specific to this product**
* *Language and accent.* The rubric includes "grammar/vocabulary". Indian English, regional idiom and code-mixing can be unfairly
  penalised; Whisper's accuracy varies by accent, and transcript errors propagate into scores.
* *Length and style bias.* On our synthetic dataset, answer length alone predicts the label about 68% of the time
  (`reports/classifier_eval.md`). A classifier trained on it can reward verbosity and punish concise, excellent answers.
  **Known failure:** a short but strong answer can be rated "average" (noted in `tests/test_classifier.py`).
* *Cultural fit of the STAR format.* STAR is a Western behavioural-interview convention; answers that are modest or collective in
  tone may score lower on "confidence".
* *Dataset provenance.* The 204 labelled answers were composed by a script from LLM-authored fragments and are **not**
  human-reviewed (`data/DATASET_CARD.md`).
* *LLM judge bias.* LLM graders can prefer longer, more formal or more "LLM-like" text, and are inconsistent across runs.

**Mitigations implemented**
* Prompts instruct the judge to evaluate content only and ignore accent, name, gender, college or region (`rubric_evaluation.v2`).
* Two independent evaluators are shown side by side; disagreement is visible rather than hidden.
* Scores are explained (reason per criterion, score breakdown, formula shown) and labelled "guidance only".
* DSA correctness is deterministic; the LLM cannot raise a failing solution's score (review marks are capped by pass-rate).
* `notebooks/classifier_training.ipynb` includes a counterfactual probe (appending different identity sentences) and a length-only
  baseline, and `scripts/eval_consistency.py` quantifies run-to-run variance.

**Not done (do these before real use):** a human-reviewed, demographically diverse evaluation set; inter-rater agreement between
human graders and the system; subgroup error analysis; an appeal/feedback channel; regular re-audits when the model or prompt changes.

## 2. Privacy of resumes and answers
* **Data minimisation.** Only what each feature needs is stored: parsed resume JSON, extracted text, scores, answers, feedback.
* **No raw resume text in logs** (tested in `tests/test_api_flow.py::test_resume_text_is_not_logged`); LLM call logs record
  provider, model, latency and task name only.
* **Third-party processing.** With a hosted LLM provider, resume excerpts and answers are sent to that provider. Use Ollama to
  keep data on the machine, check the provider's data-retention terms, and say so in your privacy notice.
* **Security basics.** bcrypt password hashes, JWT with a startup check that refuses the default secret outside `dev`, a rate limit
  on expensive endpoints, per-user data isolation (tested), file-type and size checks on uploads.
* **Admin view.** Only consenting students are included; ids are salted hashes; per-student tables and CSV export are suppressed for
  batches smaller than `ADMIN_MIN_GROUP` (default 5), a simple k-anonymity guard. Small batches can still leak by inference -
  aggregate wisely.

## 3. Transparency and explainability
Every score has a visible derivation: ATS breakdown, rubric reasons, classifier probabilities, DSA component table with the exact
formula and hint penalty, and an explanations panel in the feedback report. The LLM provider/model used for each result and
whether a rule-based fallback replaced it are displayed. A banner appears when the mock LLM or a degraded mode is active.

## 4. Consent and data deletion
* Consent is captured at signup, can be toggled at any time, and AI features refuse to run without it.
* "Export my data" returns everything stored about the user as JSON.
* "Delete my account and all data" removes profile, resumes, sessions, answers, feedback, roadmaps and progress via cascading
  foreign keys (tested). Backups outside the database are the deployer's responsibility.

## 5. India's Digital Personal Data Protection Act, 2023 (DPDP)
*This is an engineering summary, not legal advice. Check the current Act, the notified Rules and your institution's counsel.*

| DPDP concept | How the app lines up | Gap / action for a real deployment |
|---|---|---|
| Notice and consent (clear, specific, withdrawable) | Consent checkbox at signup with plain-language purpose; toggle to withdraw | Add a full privacy notice (purposes, retention, processors, grievance contact) and consent in English/Tamil/Hindi |
| Purpose limitation and data minimisation | Data used only for the feedback features | Document each purpose; avoid reusing data for model training without fresh consent |
| Right to access / correction / erasure | Export, edit profile, delete-all implemented | Add a way to correct a parsed resume; define SLAs |
| Data Fiduciary duties (security safeguards, breach notification) | Hashing, per-user isolation, Playground rate limit | Encryption at rest, access logging, breach-response runbook, vendor (LLM provider) contracts |
| Children's data (under 18) needs verifiable parental consent | Not implemented | Many users are students; add age gating or restrict to 18+ users |
| Cross-border transfer | Hosted LLMs may process data abroad | Check permitted jurisdictions; offer a local-model (Ollama) mode |
| Grievance redressal | None | Name a grievance officer and contact |

## 6. Limits of LLM scoring (read before trusting a number)
* Scores are **estimates from a stochastic model** and can vary between identical runs; `scripts/eval_consistency.py` measures it.
* Judges favour fluent text; they cannot verify facts in the answer ("I increased sales 40%").
* A rubric score is not a hiring prediction. Never use these scores to screen, rank or reject real candidates.
* Rule-based fallbacks keep the app working when the API fails, but they are crude (surface-feature heuristics).
* Translations (Tamil/Hindi) are machine-generated and unreviewed; the PDF needs a Unicode font (`PDF_FONT_PATH`).

## 7. Debates (for viva / report)

### 7.1 AI interviewers vs human interviewers
| | AI practice partner | Human interviewer |
|---|---|---|
| Availability and cost | 24/7, nearly free per session; judgement-free practice | Scarce, expensive, scheduling friction |
| Consistency | Same rubric every time *in principle* - but sampling noise and prompt drift exist | Varies by person and mood; strong halo effects |
| Depth of understanding | Pattern-based; misses context, sarcasm, true domain novelty | Can probe, read the room, assess culture and motivation |
| Bias | Can encode training-data and dataset bias at scale, invisibly | Individual biases, but accountable and challengeable |
| Feedback | Instant, detailed, repeatable | Slower, richer, relationship-building |
| Accountability | Diffuse (vendor, deployer, model) | Clear |
**Position taken by this project:** AI is a *rehearsal tool* that complements human mentors and placement officers; it must not be a
gatekeeper. High-stakes decisions stay with accountable humans.

### 7.2 Prompting vs RAG vs fine-tuning
| | Prompting | RAG | Fine-tuning (LoRA) |
|---|---|---|---|
| Data needed | None | A curated corpus | Hundreds+ of labelled examples |
| Setup cost | Minutes | Hours (index + retrieval) | Hours + GPU + evaluation |
| Updating knowledge | Edit the prompt | Edit documents | Retrain |
| Factual grounding | Weak | Strong (cites sources) | Weak unless data encodes facts |
| Style/format control | Moderate | Moderate | Strong |
| Inference cost/latency | API cost per call | API cost + retrieval | Cheap local inference |
| Privacy | Data leaves the machine (hosted) | Same | Can run fully local |
| Risks | Variance, drift between model versions | Bad retrieval = confident wrong context | Overfits templated data; bakes in dataset bias |
**How PrepWise uses them:** prompting for generation and evaluation, RAG for questions and courses (grounding + catalog
validation), and a LoRA experiment to learn what fine-tuning buys on a small synthetic dataset
(`scripts/compare_approaches.py` - including a no-model nearest-neighbour baseline that is hard to beat on templated text).
