"""Very small in-memory sliding-window rate limiter (per process).

Used to protect the Prompt Playground, which lets a logged-in user trigger arbitrary prompts
against the server's LLM API key. For multi-instance deployments use a gateway-level limiter.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException

_hits: dict[str, deque] = defaultdict(deque)


def check(key: str, max_calls: int, window_sec: int) -> None:
    """Raise HTTP 429 when ``key`` exceeded ``max_calls`` in the last ``window_sec`` seconds."""
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > window_sec:
        q.popleft()
    if len(q) >= max_calls:
        retry = int(window_sec - (now - q[0])) + 1
        raise HTTPException(429, f"Too many requests. Try again in about {retry} seconds.")
    q.append(now)


def reset() -> None:
    _hits.clear()
