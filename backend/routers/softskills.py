"""Soft-skills / HR interview: question generation, text or voice answers, follow-ups, dual evaluation."""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import require_consent
from backend.llm.runner import run_prompt
from backend.models import db_models as M
from backend.models.schemas import SoftAnswerIn, SoftQuestionsIn, TTSIn
from backend.routers.common import ensure_session_open, get_or_create_session, own_answer
from backend.services import classifier_service as C
from backend.services import evaluator as E
from backend.services import heuristics as H
from backend.services import nlp_service as N
from backend.services import speech_service as SP

router = APIRouter(prefix="/soft", tags=["softskills"])
log = logging.getLogger("prepwise.soft")


def _resume_summary(db: Session, user: M.User) -> tuple[str, list[str], list[str]]:
    r = db.scalars(select(M.Resume).where(M.Resume.user_id == user.id).order_by(M.Resume.id.desc())).first()
    if not r:
        return "No resume uploaded.", [], []
    p = r.parsed_json or {}
    projects = [x["name"] for x in p.get("projects", [])][:4]
    skills = p.get("skills", [])[:10]
    edu = [e.get("degree") or e.get("institution", "") for e in p.get("education", [])][:2]
    return (f"Skills: {', '.join(skills)}. Projects: {'; '.join(projects)}. Education: {'; '.join(edu)}.", projects, skills)


@router.post("/questions")
def questions(body: SoftQuestionsIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    session = get_or_create_session(db, user, body.session_id, "softskills")
    ensure_session_open(session)
    summary, projects, skills = _resume_summary(db, user)
    asked = [a.question_json.get("question", "") for a in db.scalars(
        select(M.Answer).where(M.Answer.user_id == user.id, M.Answer.question_type == "soft")
        .order_by(M.Answer.id.desc()).limit(20))]
    ctx = {"count": body.count, "categories": body.categories, "target_role": user.target_role,
           "company_type": user.company_type, "projects": projects, "skills": skills, "avoid": asked}
    res = run_prompt("soft_question_generation",
                     {"company_type": user.company_type, "experience_level": user.experience_level,
                      "target_role": user.target_role, "count": body.count, "categories": body.categories,
                      "resume_summary": summary, "avoid": asked[:10], "_ctx": ctx},
                     fallback=lambda: H.heuristic_soft_questions(**ctx))
    out = []
    for q in res.data.questions[: body.count]:
        row = M.Answer(session_id=session.id, user_id=user.id, question_type="soft", topic=q.category,
                       question_json=q.model_dump())
        db.add(row)
        db.flush()
        out.append({"answer_id": row.id, "question": q.question, "category": q.category})
    db.commit()
    return {"session_id": session.id, "questions": out, "llm": res.info()}


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), user: M.User = Depends(require_consent)):
    """Voice -> text with Whisper. The UI shows the transcript so the user can edit it before submitting."""
    data = await audio.read()
    return SP.transcribe(data, audio.filename or "answer.wav")  # SpeechUnavailable -> 503 (see main.py)


@router.post("/tts")
def tts(body: TTSIn, user: M.User = Depends(require_consent)):
    return Response(SP.synthesize(body.text, body.lang), media_type="audio/mpeg")


def evaluate_answer(question: str, text: str, duration_sec: float | None, segments: list[dict] | None) -> dict:
    """Run both evaluators side by side and compute analytics + overall score."""
    analytics = N.speech_analytics(text, duration_sec, segments)
    rubric, info = E.rubric_evaluate(question, text)
    try:
        cls = C.predict(text)
    except Exception as exc:  # dataset missing, sklearn mismatch...
        log.warning("classifier unavailable: %s", type(exc).__name__)
        cls = None
    rubric_100 = E.rubric_score_100(rubric)
    overall = E.compute_soft_score(rubric_100, cls["probabilities"] if cls else None, analytics)
    return {"rubric": rubric.model_dump(), "rubric_llm": info, "classifier": cls, "analytics": analytics,
            "score": overall, "rubric_100": rubric_100}


@router.post("/{answer_id}/answer")
def answer(answer_id: int, body: SoftAnswerIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    a = own_answer(db, user, answer_id, "soft")
    session = db.get(M.InterviewSession, a.session_id)
    ensure_session_open(session)
    if a.answer_text:
        raise HTTPException(409, "This question has already been answered.")
    question = a.question_json["question"]
    res = evaluate_answer(question, body.text, body.duration_sec, body.segments)
    a.answer_text = body.text
    a.evaluation = {"rubric": res["rubric"], "llm": res["rubric_llm"]}
    a.classifier = res["classifier"] or {}
    a.analytics = res["analytics"]
    a.score, a.score_breakdown = res["score"]["total"], res["score"]
    a.answered_at = datetime.now(UTC)

    # Prompt-chain step: derive a follow-up from THIS answer (max 2 per root question, enforced in code).
    fu = E.followup_question(question, body.text, a.followup_depth)
    follow = None
    if fu.needs_followup:
        root = a.followup_of or a.id
        child = M.Answer(session_id=a.session_id, user_id=user.id, question_type="soft", topic="followup",
                         question_json={"question": fu.question, "category": "followup", "reason": fu.reason},
                         followup_of=root, followup_depth=a.followup_depth + 1)
        db.add(child)
        db.flush()
        follow = {"answer_id": child.id, "question": fu.question, "reason": fu.reason, "depth": child.followup_depth,
                  "max": E.get_settings().max_followups}
    db.commit()
    return {**res, "answer_id": a.id, "followup": follow,
            "disclaimer": "Scores are guidance only (LLM rubric + small classifier + delivery heuristics)."}
