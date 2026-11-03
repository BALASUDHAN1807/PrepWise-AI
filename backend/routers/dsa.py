"""DSA / technical interview: question generation, hints, run, submit."""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import require_consent
from backend.models import db_models as M
from backend.models.schemas import CodeIn, DSAQuestionIn
from backend.routers.common import ensure_session_open, get_or_create_session, own_answer
from backend.services import evaluator as E
from backend.services import question_service as Q
from backend.services.code_runner import UnsupportedLanguage, available_languages, get_sandbox, network_isolation_mode

router = APIRouter(prefix="/dsa", tags=["dsa"])
START_DIFFICULTY = {"fresher": "easy", "junior": "medium", "mid": "medium"}


@router.get("/languages")
def languages():
    return {"available": available_languages(), "isolation": network_isolation_mode()}


def _adaptive_difficulty(db: Session, user: M.User, session: M.InterviewSession) -> tuple[str, str]:
    last = db.scalars(select(M.Answer).where(M.Answer.session_id == session.id, M.Answer.question_type == "dsa",
                                             M.Answer.score.is_not(None)).order_by(M.Answer.id.desc())).first()
    if last is None:
        return START_DIFFICULTY.get(user.experience_level, "easy"), "first question of the session"
    nxt = E.next_difficulty(last.difficulty, last.score)
    return nxt, f"previous score {last.score:.0f} on a {last.difficulty} question -> {nxt}"


def _weak_topics(db: Session, user: M.User) -> list[str]:
    rows = db.scalars(select(M.Answer).where(M.Answer.user_id == user.id, M.Answer.question_type == "dsa",
                                             M.Answer.score.is_not(None)).order_by(M.Answer.id.desc()).limit(20))
    return sorted({a.topic for a in rows if a.score < 60})[:3]


@router.post("/question")
def new_question(body: DSAQuestionIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    session = get_or_create_session(db, user, body.session_id, "dsa")
    ensure_session_open(session)
    difficulty, why = (body.difficulty, "chosen by user") if body.difficulty else _adaptive_difficulty(db, user, session)
    resume = db.scalars(select(M.Resume).where(M.Resume.user_id == user.id).order_by(M.Resume.id.desc())).first()
    skills = (resume.parsed_json or {}).get("skills", [])[:12] if resume else []
    recent = [a.question_json.get("title", "") for a in db.scalars(
        select(M.Answer).where(M.Answer.user_id == user.id, M.Answer.question_type == "dsa")
        .order_by(M.Answer.id.desc()).limit(10))]
    gen = Q.safe_generate_question(topic=body.topic, difficulty=difficulty, role=user.target_role,
                                   company_style=body.company_style or user.company_type, skills=skills,
                                   weak_areas=_weak_topics(db, user), avoid_titles=recent)
    q = gen["question"]
    ans = M.Answer(session_id=session.id, user_id=user.id, question_type="dsa", topic=body.topic,
                   difficulty=difficulty, question_json=q)
    db.add(ans)
    db.commit()
    db.refresh(ans)
    return {"answer_id": ans.id, "session_id": session.id, "question": Q.public_view(q),
            "difficulty": difficulty, "difficulty_reason": why, "generation": gen["meta"]}


@router.post("/{answer_id}/hint")
def hint(answer_id: int, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    a = own_answer(db, user, answer_id, "dsa")
    hints = a.question_json.get("hints", [])
    if a.hints_used >= len(hints):
        raise HTTPException(409, "All hints have already been used.")
    a.hints_used += 1
    db.commit()
    return {"hint_number": a.hints_used, "hint": hints[a.hints_used - 1], "hints_remaining": len(hints) - a.hints_used,
            "penalty_per_hint": E.get_settings().hint_penalty}


@router.post("/{answer_id}/run")
def run(answer_id: int, body: CodeIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    """Run against the VISIBLE tests (and optional custom stdin). Nothing is scored or stored."""
    a = own_answer(db, user, answer_id, "dsa")
    ensure_session_open(db.get(M.InterviewSession, a.session_id))
    q = a.question_json
    lang = "sql" if q.get("kind") == "sql" else body.language
    sb = get_sandbox()
    try:
        visible = [t for t in q["test_cases"] if not t.get("hidden")]
        results = sb.run_tests(lang, body.code, visible, q.get("kind", "code"))
        custom = None
        if body.stdin is not None:
            r = sb.run_once(lang, body.code, body.stdin)
            custom = {"stdout": r.stdout, "stderr": r.stderr[-1500:], "timed_out": r.timed_out,
                      "runtime_ms": round(r.runtime_ms, 1)}
    except UnsupportedLanguage as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"summary": E.summarize_cases(results), "custom": custom}


@router.post("/{answer_id}/submit")
def submit(answer_id: int, body: CodeIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    a = own_answer(db, user, answer_id, "dsa")
    session = db.get(M.InterviewSession, a.session_id)
    ensure_session_open(session)
    q = a.question_json
    lang = "sql" if q.get("kind") == "sql" else body.language
    try:
        res = E.evaluate_submission(q, lang, body.code, hints_used=a.hints_used)
    except UnsupportedLanguage as exc:
        raise HTTPException(400, str(exc)) from exc
    a.code, a.language = body.code, lang
    a.test_results = res["tests"]
    a.evaluation = {"review": res["review"], "llm": res["review_llm"]}
    a.score, a.score_breakdown = res["score"]["total"], res["score"]
    a.answered_at = datetime.now(UTC)
    db.commit()
    return {**res, "answer_id": a.id, "next_difficulty": E.next_difficulty(a.difficulty, a.score),
            "reference_solution": q["reference_solution"] if a.hints_used >= 3 or a.score >= 100 else None,
            "disclaimer": "Correctness comes from executing test cases; the LLM review of logic and style is guidance only."}
