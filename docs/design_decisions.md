# Design decisions

## 1. Pipelines and RAG, not agents
The brief forbids autonomous agents, and it is also the better engineering choice for this product:

* **Predictability.** Every user action maps to a fixed sequence of function calls. The feedback chain is always
  `summary -> skill-gap mapping -> per-question feedback -> improvement plan`; a follow-up question is *one* prompt triggered by
  *our* code and capped at 2 per question. Nothing decides its own next step or calls tools in a loop.
* **Testability.** Each step has a schema and a fallback, so unit tests can cover it (`tests/`).
* **Cost and latency.** A bounded number of calls per request; no open-ended loops.
* **Safety.** An agent with a code-execution tool in a student-facing app is a bigger attack surface than a sandbox we invoke ourselves.

Where "agent-like" behaviour was tempting we replaced it with code: adaptive difficulty is `next_difficulty(score)`; the hallucination
guard is "run the reference solution against the tests, regenerate up to 3 times, else serve a verified bank question".

**RAG** grounds generation in data we control: similar questions from `question_bank.json` (style + fewer hallucinations) and
courses from `courses.json` (the model may only choose retrieved ids and the code rejects any id not in the catalog).

## 2. One provider-agnostic LLM wrapper
`backend/llm/client.py` exposes `complete()` and `complete_json()` over Gemini, OpenAI, Anthropic and Ollama (plain `httpx`, no
vendor SDK lock-in). Provider and model come from `.env`. `complete_json` validates against a Pydantic schema, retries up to 2 times
with the validation error fed back, and finally returns a *rule-based fallback* (flagged `used_fallback`, shown in the UI) so a
flaky API never produces a stack trace. A `mock` provider (rule-based) powers CI, offline demos and tests.

## 3. Prompts are data
Each prompt is a versioned YAML file (`backend/llm/prompts/<name>.v<N>.yaml`) with system/user templates (Jinja2 `StrictUndefined`
so a missing variable fails loudly), a temperature (0.2 for evaluation, 0.7 for generation) and an output schema name. The Prompt
Playground edits and compares exactly these templates.

## 4. Correctness comes from execution, not opinion
Code is scored by running it. `compute_dsa_score` = 60% tests + 20% logic + 10% complexity + 10% quality - 5/hint, and the LLM's
logic/complexity/quality marks are **capped by the test pass-rate** (an LLM cannot award 10/10 to code that fails every test).
The sandbox: subprocess per test, CPU/memory/file-size rlimits, process-group kill on timeout, clean environment, network namespace
(`unshare -rn`) when the host allows it plus an in-process socket guard otherwise. **It is not a hardened multi-tenant sandbox** -
for a public deployment use `CODE_RUNNER=judge0` or a container/gVisor runner (see README limitations).

## 5. Model choices
| Need | Choice | Why | Trade-off |
|---|---|---|---|
| Text generation / evaluation | Configurable hosted LLM (default Gemini; also OpenAI/Anthropic/Ollama) | Quality, JSON mode, student-friendly latency | Network + privacy (answer text leaves the machine); Ollama keeps data local |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Small (≈90 MB), CPU-fast, good for short skill/question text | English-centric; deterministic hashing fallback when weights cannot be downloaded |
| Vector store | ChromaDB (persistent), in-memory fallback | Zero-ops, local | Not for large multi-tenant corpora |
| NLP | spaCy `en_core_web_sm` + PhraseMatcher on the skills taxonomy | Fast, transparent skill matching + NER | Skill list is hand-curated; blank-pipeline fallback loses NER |
| Speech | `faster-whisper` (base, int8) | CPU-friendly, offline, free | Accuracy on accented speech varies; larger models are slower |
| Discriminative classifier | TF-IDF + structural features + Logistic Regression | Tiny, inspectable, fast; honest baselines exist | Learns surface cues on synthetic data; optional DistilBERT adds capacity but needs training |
| Fine-tuning experiment | Flan-T5-small + LoRA | Runs on CPU/free Colab; well under 2% trainable params | Small model; synthetic templated targets |
| DB | SQLite via SQLAlchemy | No setup for students | Single-writer; swap `DATABASE_URL` for Postgres in production |
| UI | Streamlit multi-page | Low-code (Unit 3), fast iteration | Not a polished product front-end; timers use fragments |
| API | FastAPI | Typed, OpenAPI docs for free | - |

## 6. Privacy by design
Consent gate (`require_consent`) on every AI feature; one-click cascade deletion (FK `ON DELETE CASCADE`); raw resume text never
logged; admin dashboard sees only consenting students, salted-hash ids, aggregates, and k-anonymity suppression
(`ADMIN_MIN_GROUP`, default 5). See `docs/responsible_ai.md`.

## 7. Things we deliberately did not do
* No scraping of LeetCode, HackerRank, GfG, CodeChef, LinkedIn, Indeed, Glassdoor, Coursera/Udemy/NPTEL pages. Questions are
  original (`scripts/build_question_bank.py`); course entries are a small manual catalog of real course pages.
* No auto-grading claims beyond "guidance"; no hiring decisions.
* No GAN/VAE (see `docs/generative_vs_discriminative.md`).
