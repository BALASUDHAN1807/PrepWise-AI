"""Sessions (incl. timed mock interviews), feedback generation, dashboard and PDF report."""
from __future__ import annotations

import copy
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import get_current_user, require_consent
from backend.models import db_models as M
from backend.models.schemas import FeedbackIn, SessionIn
from backend.routers.common import seconds_remaining
from backend.services import feedback_service as F
from backend.services import report_service as RP

router = APIRouter(tags=["feedback"])


def _own_session(db: Session, user: M.User, session_id: int) -> M.InterviewSession:
    s = db.get(M.InterviewSession, session_id)
    if not s or s.user_id != user.id:
        raise HTTPException(404, "Session not found.")
    return s


def _session_view(db: Session, s: M.InterviewSession) -> dict:
    answers = db.scalars(select(M.Answer).where(M.Answer.session_id == s.id).order_by(M.Answer.id)).all()
    return {"id": s.id, "kind": s.kind, "status": s.status, "started_at": s.started_at.isoformat(),
            "time_limit_min": s.time_limit_min, "seconds_remaining": seconds_remaining(s),
            "overall_score": s.overall_score, "meta": s.meta,
            "questions": [{"answer_id": a.id, "type": a.question_type, "topic": a.topic,
                           "title": a.question_json.get("title") or a.question_json.get("question", ""),
                           "answered": a.score is not None, "score": a.score, "followup_depth": a.followup_depth}
                          for a in answers]}


@router.post("/sessions")
def start_session(body: SessionIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    """Start a session. ``kind='mock'`` creates a timed DSA round + HR round."""
    limit = body.time_limit_min if body.kind == "mock" else None
    if body.kind == "mock" and limit is None:
        limit = 45
    s = M.InterviewSession(user_id=user.id, kind=body.kind, time_limit_min=limit,
                           meta={"rounds": ["dsa", "hr"], "dsa_count": body.dsa_count, "hr_count": body.hr_count}
                           if body.kind == "mock" else {})
    db.add(s)
    db.commit()
    db.refresh(s)
    return _session_view(db, s)


@router.get("/sessions")
def list_sessions(user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(M.InterviewSession).where(M.InterviewSession.user_id == user.id)
                      .order_by(M.InterviewSession.id.desc()).limit(50)).all()
    return [{"id": s.id, "kind": s.kind, "status": s.status, "started_at": s.started_at.isoformat(),
             "overall_score": s.overall_score,
             "has_feedback": db.scalar(select(M.Feedback.id).where(M.Feedback.session_id == s.id)) is not None}
            for s in rows]


@router.get("/sessions/{session_id}")
def get_session(session_id: int, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _session_view(db, _own_session(db, user, session_id))


@router.post("/feedback/{session_id}")
def generate(session_id: int, body: FeedbackIn | None = None, user: M.User = Depends(require_consent),
             db: Session = Depends(get_db)):
    """Run the feedback prompt chain for a session and store it (English + requested language)."""
    s = _own_session(db, user, session_id)
    try:
        fb = F.generate_feedback(db, user, s, (body.language if body else None))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    s.ended_at = datetime.now(UTC)
    db.commit()
    return {"feedback_id": fb.id, "session_id": s.id, "language": fb.language, "content": fb.content}


def _latest_feedback(db: Session, user: M.User, session_id: int) -> M.Feedback:
    _own_session(db, user, session_id)
    fb = db.scalars(select(M.Feedback).where(M.Feedback.session_id == session_id, M.Feedback.user_id == user.id)
                    .order_by(M.Feedback.id.desc())).first()
    if not fb:
        raise HTTPException(404, "No feedback yet for this session. Generate it first.")
    return fb


@router.get("/feedback/{session_id}")
def get_feedback(session_id: int, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    fb = _latest_feedback(db, user, session_id)
    return {"feedback_id": fb.id, "language": fb.language, "content": fb.content, "content_en": fb.content_en,
            "created_at": fb.created_at.isoformat()}


@router.post("/feedback/{session_id}/translate")
def translate(session_id: int, body: FeedbackIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    """Show the stored feedback in another language without re-running the whole chain."""
    s = _own_session(db, user, session_id)
    base = _latest_feedback(db, user, session_id)
    lang = body.language or "en"
    content = copy.deepcopy(base.content_en)
    if lang != "en":
        paths = F._collect_strings(content)
        translated, ok = F._translate([t for _, t in paths], lang, None)
        for (path, _), new in zip(paths, translated):
            F._set_path(content, path, new)
        content["meta"]["translation"] = {"language": lang, "ok": ok}
        if not ok:
            content["meta"]["translation"]["note"] = "Translation unavailable with this provider; showing English."
    content["language"] = lang
    fb = M.Feedback(session_id=s.id, user_id=user.id, language=lang, content=content, content_en=base.content_en)
    db.add(fb)
    db.commit()
    return {"feedback_id": fb.id, "language": lang, "content": content}


@router.get("/feedback/{session_id}/report.pdf")
def report_pdf(session_id: int, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = _own_session(db, user, session_id)
    fb = _latest_feedback(db, user, session_id)
    answers = db.scalars(select(M.Answer).where(M.Answer.session_id == s.id)).all()
    pdf = RP.build_pdf(user, s, fb, answers)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="prepwise-report-{s.id}.pdf"'})


@router.get("/dashboard")
def dashboard(user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return F.dashboard(db, user)
