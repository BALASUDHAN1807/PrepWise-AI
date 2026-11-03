"""Central application settings.

Every tunable value (LLM provider, model names, temperatures, sandbox limits,
embedding backend, scoring weights) is read from environment variables or a
``.env`` file so the same code runs locally, in CI and on Hugging Face Spaces.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed settings loaded from the environment (see ``.env.example``)."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---------------------------------------------------------------
    app_name: str = "PrepWise AI"
    environment: str = "dev"
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'prepwise.db'}"
    jwt_secret: str = "change-me-in-production"
    jwt_expire_minutes: int = 720
    admin_emails: str = ""  # comma-separated emails that become admins on signup
    anon_salt: str = "prepwise-anon-salt"  # used to anonymise ids in admin exports
    backend_url: str = "http://localhost:8000"  # used by the Streamlit frontend

    # --- LLM (one provider-agnostic wrapper) -------------------------------
    llm_provider: str = "gemini"  # gemini | openai | anthropic | ollama | mock
    llm_model: str = ""  # optional override applied to whichever provider is active
    llm_timeout_sec: int = 90
    llm_max_retries: int = 2  # retries after the first attempt on invalid JSON / errors
    temperature_eval: float = 0.2  # evaluation and scoring prompts
    temperature_gen: float = 0.7  # question generation prompts

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- Embeddings / RAG --------------------------------------------------
    embedding_backend: str = "sentence-transformers"  # sentence-transformers | hash
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store: str = "chroma"  # chroma | memory
    chroma_dir: str = str(PROJECT_ROOT / ".chroma")

    # --- NLP / speech ------------------------------------------------------
    spacy_model: str = "en_core_web_sm"
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    enable_transformer_sentiment: bool = True
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"

    # --- Code sandbox ------------------------------------------------------
    code_runner: str = "local"  # local | judge0
    sandbox_timeout_sec: float = 4.0
    sandbox_memory_mb: int = 256
    sandbox_compile_timeout_sec: float = 30.0
    judge0_url: str = "https://judge0-ce.p.rapidapi.com"
    judge0_api_key: str = ""
    judge0_api_host: str = "judge0-ce.p.rapidapi.com"

    pdf_font_path: str = ""  # optional Unicode TTF (e.g. NotoSansTamil/NotoSansDevanagari) for Tamil/Hindi PDFs
    max_audio_mb: int = 15
    tts_enabled: bool = True

    # --- Scoring -----------------------------------------------------------
    hint_penalty: float = 5.0
    readiness_weight_resume: float = 0.2
    readiness_weight_dsa: float = 0.4
    readiness_weight_soft: float = 0.4
    max_followups: int = 2
    admin_min_group: int = 5  # k-anonymity: hide per-student views / exports for batches smaller than this

    # --- Paths -------------------------------------------------------------
    data_dir: str = str(PROJECT_ROOT / "data")
    models_dir: str = str(PROJECT_ROOT / "models_store")
    prompts_dir: str = str(PROJECT_ROOT / "backend" / "llm" / "prompts")

    @property
    def admin_email_set(self) -> set[str]:
        """Lower-cased admin emails parsed from the comma separated setting."""
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
