# Deploying to Hugging Face Spaces (and using Hugging Face models)

*Syllabus link: the new AI stack - GitHub (code + CI), Kaggle-style dataset card, Hugging Face models and Spaces.*

> The files in `deploy/hf_space/` were written from Hugging Face's documented Docker-Space conventions, but **could not be built or
> deployed from the authoring environment** (no Docker daemon, no access to huggingface.co). Expect to fix small things on first deploy.

## Why a Docker Space
PrepWise is two processes (FastAPI + Streamlit) plus Java/C++ toolchains for the code sandbox. A Docker Space can run all of that
in one container; Streamlit is exposed on port 7860 and talks to the API on `127.0.0.1:8000`.

## Steps
1. **Create the Space** at huggingface.co -> New Space -> SDK **Docker** -> Blank. Choose CPU Basic (free) to start.
2. **Add the code**: clone the Space repo and copy this project into it, then
   ```bash
   cp deploy/hf_space/Dockerfile ./Dockerfile
   cp deploy/hf_space/SPACE_README.md ./README.md      # keeps the YAML front-matter (sdk: docker, app_port: 7860)
   git add . && git commit -m "Deploy PrepWise AI" && git push
   ```
   (Keep the project's own README under another name such as `README_PROJECT.md`, or deploy from a branch.)
3. **Set secrets** in Space -> Settings -> *Variables and secrets* (never commit them):
   `LLM_PROVIDER` (e.g. `gemini`), `GEMINI_API_KEY` (or the key for your provider), `JWT_SECRET` (random 64 hex chars),
   `ADMIN_EMAILS`, `ANON_SALT`. Optional: `ADMIN_MIN_GROUP`, `WHISPER_MODEL`.
4. **Wait for the build** (several minutes: PyTorch, spaCy model, MiniLM download). Open the Space; the sidebar shows which LLM and
   fallbacks are active - if it says *Mock LLM*, your secrets are missing.
5. **Persistence (important):** the free tier's disk is ephemeral, so the SQLite DB resets on restart. For real use attach
   persistent storage and set `DATABASE_URL=sqlite:////data/prepwise.db`, `CHROMA_DIR=/data/chroma`, `MODELS_DIR=/data/models`, or point
   `DATABASE_URL` at an external Postgres.

## Known constraints on Spaces
* **Code sandbox.** `unshare -rn` (network isolation) needs user namespaces, which are usually unavailable; the sandbox then falls back
  to an in-process socket guard and `/health` says so. Do not expose a public Space to untrusted users with arbitrary code execution: use
  `CODE_RUNNER=judge0` or disable the DSA page.
* **Memory.** Free CPU Spaces have ~16 GB RAM / 2 vCPU: use `WHISPER_MODEL=tiny` or `base`; voice transcription will be slow.
* **Privacy.** A public Space + hosted LLM means user text goes to that provider and the Space host. Add your privacy notice and keep the
  consent gate on.
* **Fonts.** For Tamil/Hindi PDF reports add a Unicode TTF (Noto Sans Tamil/Devanagari) to the image and set `PDF_FONT_PATH`.

## Publishing models and data to the Hub
* **LoRA adapter:** after `python scripts/finetune_lora.py`, upload `models/lora_flan_t5_small/` with `huggingface-cli upload <user>/prepwise-feedback-lora models/lora_flan_t5_small .`
  and use `docs/model_card.md` (section A) as the repo README. State the base model licence.
* **Dataset:** upload `data/soft_skill_labeled.csv` / `feedback_sft.jsonl` as a dataset repo (or Kaggle) with `data/DATASET_CARD.md`.
  Keep the "synthetic, unreviewed" warnings prominent.
* **Consuming Hub models** (already supported): `EMBEDDING_MODEL`, `WHISPER_MODEL`, `SENTIMENT_MODEL` env vars; a fine-tuned DistilBERT saved
  under `<MODELS_DIR>/distilbert_softskill` is picked up automatically (`scripts/train_distilbert_classifier.py`).

## GitHub side (CI)
`.github/workflows/ci.yml` runs lint + tests on Python 3.11/3.12 and executes both notebooks in offline smoke mode on every push/PR.
