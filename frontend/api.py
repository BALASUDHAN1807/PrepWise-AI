"""Thin HTTP client + shared UI helpers for the Streamlit pages.

All business logic lives in the FastAPI backend; the frontend only renders and calls it.
The JWT is kept in ``st.session_state`` (per browser tab) - never written to disk.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # same .env as the backend (BACKEND_URL etc.)
API_URL = (os.getenv("BACKEND_URL") or os.getenv("API_URL") or "http://localhost:8000").rstrip("/")
DISCLAIMER = ("Scores are AI-assisted guidance, not a hiring decision. Code correctness comes only from running "
              "test cases; essay-style scores come from an LLM rubric plus a small classifier and can be wrong.")


class ApiError(Exception):
    """Backend returned an error; ``str(exc)`` is a user-presentable message."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _detail(resp: httpx.Response) -> str:
    try:
        d = resp.json().get("detail", resp.text)
    except Exception:
        return resp.text[:300] or f"HTTP {resp.status_code}"
    if isinstance(d, list):  # FastAPI validation errors
        return "; ".join(f"{'.'.join(map(str, e.get('loc', [])[1:]))}: {e.get('msg', '')}" for e in d)
    return str(d)


def call(method: str, path: str, *, json: Any = None, files: Any = None, timeout: float = 180, raw: bool = False):
    """Call the backend. Returns parsed JSON (or bytes when ``raw``). Raises :class:`ApiError`."""
    headers = {}
    if token := st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.request(method, f"{API_URL}{path}", json=json, files=files, headers=headers, timeout=timeout)
    except httpx.ConnectError as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with: uvicorn backend.main:app --port 8000") from exc
    except httpx.TimeoutException as exc:
        raise ApiError("The request timed out - the AI provider may be slow. Try again.") from exc
    if resp.status_code == 401:
        st.session_state.pop("token", None)
        st.session_state.pop("user", None)
        raise ApiError("Your session expired. Please log in again on the Home page.", 401)
    if resp.status_code >= 400:
        raise ApiError(_detail(resp), resp.status_code)
    return resp.content if raw else resp.json()


def get(path: str, **kw):
    return call("GET", path, **kw)


def post(path: str, body: Any = None, **kw):
    return call("POST", path, json=body, **kw)


def try_call(fn, *args, spinner: str | None = None, **kw):
    """Run an API call, show errors in the UI and return None on failure."""
    try:
        if spinner:
            with st.spinner(spinner):
                return fn(*args, **kw)
        return fn(*args, **kw)
    except ApiError as exc:
        st.error(str(exc))
        return None


def refresh_user() -> dict | None:
    user = try_call(get, "/auth/me")
    if user:
        st.session_state["user"] = user
    return user


def sidebar_status() -> None:
    """Sidebar: who is logged in + which AI/fallback modes are active (so nothing is silently degraded)."""
    with st.sidebar:
        user = st.session_state.get("user")
        if user:
            st.caption(f"Signed in as **{user['name']}**")
            if st.button("Log out", width="stretch"):
                for k in list(st.session_state):
                    del st.session_state[k]
                st.rerun()
        try:
            h = httpx.get(f"{API_URL}/health", timeout=5).json()
        except Exception:
            st.error("API offline")
            return
        llm = h["llm"]
        st.caption(f"LLM: `{llm['provider']}` / `{llm['model']}`" + ("" if llm["ready"] else " - **API key missing**"))
        if llm["provider"] == "mock":
            st.warning("Mock LLM (rule-based). Set LLM_PROVIDER and a key in .env for real AI feedback.", icon="⚠️")
        emb = h["embeddings"]
        if not emb.get("semantic"):
            st.caption("Embeddings: hashing fallback (no semantic model)")
        if h["nlp"]["spacy_mode"] != "model":
            st.caption("spaCy: blank pipeline (no NER)")


def require_login(consent: bool = True) -> dict:
    """Stop the page unless the user is logged in (and, optionally, has given consent)."""
    st.set_page_config(page_title="PrepWise AI", page_icon="🎯", layout="wide")
    sidebar_status()
    if not st.session_state.get("token"):
        st.info("Please log in or sign up on the **Home** page first.")
        st.stop()
    user = st.session_state.get("user") or refresh_user()
    if not user:
        st.stop()
    if consent and not user.get("consent_given"):
        st.warning("You have not consented to data processing. Enable consent on the Home page to use this feature.")
        st.stop()
    return user


def disclaimer() -> None:
    st.caption(f"ℹ️ {DISCLAIMER}")


def llm_note(info: dict | None) -> None:
    """Small caption showing which model produced a result and whether the safe fallback was used."""
    if not info:
        return
    txt = f"{info.get('provider')}/{info.get('model')} · {info.get('latency_ms')} ms · attempts {info.get('attempts')}"
    if info.get("used_fallback"):
        st.caption(f"⚠️ Rule-based fallback used ({info.get('error') or 'invalid model output'}). {txt}")
    else:
        st.caption(txt)
