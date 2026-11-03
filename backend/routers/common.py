"""Shared helpers for routers: session lookup, ownership checks, timed-session enforcement."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.models import db_models as M


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def session_deadline(s: M.InterviewSession) -> datetime | None:
    return _aware(s.started_at) + timedelta(minutes=s.time_limit_min) if s.time_limit_min else None


def seconds_remaining(s: M.InterviewSession) -> int | None:
    d = session_deadline(s)
    return None if d is None else max(0, int((d - datetime.now(UTC)).total_seconds()))


def ensure_session_open(s: M.InterviewSession) -> None:
    """Reject answers after the deadline of a timed (mock) session or after completion."""
    if s.status == "completed":
        raise HTTPException(409, "This session is already completed. Start a new session.")
    rem = seconds_remaining(s)
    if rem is not None and rem <= 0:
        raise HTTPException(409, "Time is up for this mock interview. Generate your feedback report.")


def get_or_create_session(db: Session, user: M.User, session_id: int | None, kind: str) -> M.InterviewSession:
    """Return the user's session (mock sessions accept both dsa and softskills work)."""
    if session_id is None:
        s = M.InterviewSession(user_id=user.id, kind=kind)
        db.add(s)
        db.commit()
        db.refresh(s)
        return s
    s = db.get(M.InterviewSession, session_id)
    if not s or s.user_id != user.id:
        raise HTTPException(404, "Session not found.")
    if s.kind not in (kind, "mock"):
        raise HTTPException(400, f"This is a {s.kind} session; it cannot be used for {kind} questions.")
    return s


def own_answer(db: Session, user: M.User, answer_id: int, qtype: str) -> M.Answer:
    a = db.get(M.Answer, answer_id)
    if not a or a.user_id != user.id or a.question_type != qtype:
        raise HTTPException(404, "Question not found.")
    return a
