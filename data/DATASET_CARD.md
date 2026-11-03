# Dataset card: PrepWise AI data (Kaggle-style)

Everything under `data/` is **small, hand-built or script-built seed data for a course project**. It is good enough to make the app work
and to demonstrate methods; it is **not** a benchmark and has **not** been validated against real interviewers.

## Files at a glance
| File | Rows | What it is | Review status |
|---|---|---|---|
| `soft_skill_labeled.csv` | 204 | Interview questions + candidate-style answers labelled poor / average / good | **Not human-reviewed** |
| `feedback_sft.jsonl` | 204 | Synthetic structured feedback paired with each labelled answer (LoRA training data) | **Not human-reviewed** (`reviewed: false`) |
| `question_bank.json` | 33 | Original coding/SQL questions with >=6 tests, hints and reference solutions | Outputs machine-verified by running the reference solutions; wording not externally reviewed |
| `courses.json` | 28 | Catalog of real online courses (Coursera 17, NPTEL 7, Udemy 4) | URLs checked when the catalog was assembled; links rot - re-verify |
| `skills_taxonomy.json` | 28 | Skill ids, names, technical/soft category, keywords | Hand-curated |
| `filler_words.txt` | ~15 | Filler words/phrases for speech analytics | Hand-curated |

---
## 1. `soft_skill_labeled.csv`
**Purpose.** Train and evaluate the TF-IDF + Logistic Regression answer-quality classifier (and the optional DistilBERT variant).

**Columns.** `id`, `question_id`, `category` (behavioral 68, hr 51, situational 51, resume 34), `question`, `answer`, `label`
(average 72, poor 72, good 60), `source`, `reviewed` (always `no`).

**How it was made.** `scripts/generate_soft_skill_dataset.py` (seed 42) composes answers from LLM-authored sentence fragments
(written by Claude for this project) for **12 questions**, at three quality levels, and adds 24 *length-matched distractor* answers
(long-but-poor, short-but-good-ish) to weaken the length shortcut. No real students' answers, no scraped content, no personal data.

**Known problems (please read).**
* *Synthetic and templated.* Answers share fragments and phrasing; real answers are messier. Models can learn the generator's habits.
* *Only 12 questions.* Question-grouped evaluation therefore has 3 test questions (51 answers) - estimates are noisy.
* *Length confound.* Mean length: poor 18 words, average 39, good 72 - length alone reaches ~68% accuracy (leave-questions-out).
  Padded distractors reduce but do not remove this. **Known failure:** short but strong answers are under-rated.
* *Label validity.* "Good/average/poor" is defined by the generator's recipe, not by independent human judgement. No inter-rater agreement exists.
* *Narrow demographics and register.* English only; campus-placement context; no code-mixed or dialect answers.

**Suggested uses.** Teaching classical NLP evaluation (baselines, leakage, grouped CV); demonstrating fairness probes.
**Do not use for:** evaluating real candidates, deployment decisions, or claims about interview-grading accuracy.

**To make it trustworthy:** have 2-3 human reviewers (placement officers) relabel a stratified sample, compute inter-rater agreement,
add real (consented, anonymised) answers, and set `reviewed=yes` per row.

## 2. `feedback_sft.jsonl`
**Purpose.** Supervised fine-tuning targets for LoRA feedback generation (`scripts/finetune_lora.py`).

**Fields.** `id`, `question_id`, `category`, `question`, `answer`, `label`, `feedback`, `reviewed` (bool, all `false`), `source`.

**Format of `feedback`.**
`Verdict: <label>. Evidence: <N> words; STAR parts found: <list|none>; <includes|no> measurable detail. Strength: <sentence>. Improve <criterion>: <sentence>. Next step: <sentence>.`

**How it was made.** `scripts/build_feedback_dataset.py` - deterministic composition from the CSV label and the rule-based rubric in
`backend/services/heuristics.py` (no LLM). Every claim in the text is checkable against measurable features, which is how
`scripts/_feedback_eval.py` audits generated feedback.

**Limits.** Only ~154 distinct strings; templated sentences; the "weakest criterion" is the heuristic rubric's opinion (it picks
"depth" for most answers); the same heuristics generate targets and evaluate outputs, so metrics can reward template memorisation.
Treat results as a pipeline demonstration, not as evidence of better teaching. Use `--only-reviewed` once humans have reviewed rows.

## 3. `question_bank.json`
**Purpose.** RAG grounding for question generation and the verified fallback when generation fails the hallucination guard.

**Content.** 33 questions: arrays 6, dp 6, strings 5, graphs 5, trees 4, SQL 4, linked lists 3; difficulty easy 11 / medium 19 / hard 3.
Each has statement, constraints, input/output format, examples, 3 progressive hints, expected complexity, a reference solution and
**>=6 test cases** (the first 2 visible, the rest hidden).

**Provenance.** Written for this project (`scripts/build_question_bank.py`); **not** copied from LeetCode, HackerRank, GeeksforGeeks,
CodeChef or any other site. Expected outputs are produced by executing the reference solutions in the sandbox, so tests agree with the solutions
(correctness of the *problem statement* itself is not independently verified). Re-run the script to regenerate identically.

**Limits.** Small and skewed to medium difficulty; solutions are Python; classic patterns may resemble well-known problems in the
way all textbook problems do.

## 4. `courses.json`, `skills_taxonomy.json`, `filler_words.txt`
* **Courses:** id, title, platform, skills, level, URL, provider. Only these ids can appear in a roadmap; the code rejects any other id.
  Prices, ratings and availability change - only stable fields are stored. No page content was scraped.
* **Skills taxonomy:** the vocabulary that links resume skills, interview weaknesses and courses. Coverage is opinionated and incomplete.
* **Fillers:** English-only; "like"/"actually" are legitimate words in context, so counts are an estimate.

## Licensing and citation
Code and original data: choose your project licence before publishing (e.g. MIT for code, CC BY 4.0 for data). Course titles/URLs
belong to their providers and are used only as pointers. Suggested citation: *PrepWise AI seed datasets (course project), 2026.*
