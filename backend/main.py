"""FastAPI application entry point.

Run:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.database import init_db
from backend.llm.client import LLMClient, LLMError
from backend.routers import admin, auth, dsa, feedback, prompts, resume, roadmap, softskills
from backend.services import classifier_service, code_runner, nlp_service, rag_service
from backend.services.resume_service import ResumeError
from backend.services.speech_service import SpeechUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("prepwise")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    s = get_settings()
    if s.jwt_secret == "change-me-in-production":
        if s.environment != "dev":
            raise RuntimeError("Set a strong JWT_SECRET in .env before running outside dev "
                               "(python -c \"import secrets; print(secrets.token_hex(32))\").")
        log.warning("Using the default JWT secret - fine for local development only.")
    init_db()
    yield


app = FastAPI(title="PrepWise AI API", version="1.0.0", lifespan=lifespan,
              description="Personalized interview preparation and career guidance. "
                          "Fixed pipelines + RAG + structured JSON; no autonomous agents.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                   allow_methods=["*"], allow_headers=["*"])

for r in (auth, resume, dsa, softskills, feedback, roadmap, prompts, admin):
    app.include_router(r.router)


# ---- graceful error handling: user-facing messages, never stack traces or resume text ----
@app.exception_handler(LLMError)
async def llm_error(_: Request, exc: LLMError):
    log.warning("LLM error: %s", str(exc)[:200])
    return JSONResponse(status_code=502, content={
        "detail": "The AI provider is unavailable or returned unusable output. Check your API key in .env or "
                  f"try again. ({str(exc)[:160]})"})


@app.exception_handler(ResumeError)
async def resume_error(_: Request, exc: ResumeError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(SpeechUnavailable)
async def speech_error(_: Request, exc: SpeechUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(code_runner.UnsupportedLanguage)
async def lang_error(_: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health", tags=["meta"])
def health():
    """Capability report shown in the UI sidebar (so users know when a fallback is active)."""
    s = get_settings()
    try:
        llm = LLMClient()
        provider, model = llm.provider_name, llm.model
    except LLMError as exc:
        provider, model = f"misconfigured: {exc}", ""
    keys = {"gemini": s.gemini_api_key, "openai": s.openai_api_key, "anthropic": s.anthropic_api_key}
    return {
        "status": "ok",
        "llm": {"provider": provider, "model": model,
                "ready": provider in ("mock", "ollama") or bool(keys.get(provider))},
        "sandbox": {"runner": s.code_runner, "languages": code_runner.available_languages(),
                    "isolation": code_runner.network_isolation_mode()},
        "embeddings": rag_service.embedding_info(),
        "nlp": nlp_service.nlp_status(),
        "classifier": {"model_cached": (s.models_path / classifier_service.MODEL_FILE).exists()},
    }
