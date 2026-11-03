"""Resume upload, parsing, ATS score, JD match, bullet rewriter and cover letter."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import require_consent
from backend.models import db_models as M
from backend.models import llm_schemas as S
from backend.models.schemas import BulletsIn, CoverLetterIn, JDIn
from backend.services import resume_service as RS

router = APIRouter(prefix="/resume", tags=["resume"])
MAX_UPLOAD = 5 * 1024 * 1024


def _latest(db: Session, user: M.User) -> M.Resume:
    r = db.scalars(select(M.Resume).where(M.Resume.user_id == user.id).order_by(M.Resume.id.desc())).first()
    if not r:
        raise HTTPException(404, "Upload your resume first.")
    return r


@router.post("/upload")
async def upload(file: UploadFile = File(...), user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is larger than 5 MB.")
    text = RS.extract_text(file.filename or "resume.pdf", data)  # ResumeError -> 422
    parsed, info = RS.parse_resume(text)
    ats = RS.ats_score(text, parsed, user.target_role)
    sugg = RS.ats_suggestions(ats, parsed, user.target_role)
    row = M.Resume(user_id=user.id, filename=(file.filename or "resume")[:255], raw_text=text,
                   parsed_json=parsed.model_dump(), ats_score=ats["total"], ats_breakdown=ats)
    db.add(row)
    db.commit()
    return {"resume_id": row.id, "parsed": row.parsed_json, "ats": ats, "suggestions": sugg, "parse_info": info}


@router.get("/latest")
def latest(user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    r = _latest(db, user)
    return {"resume_id": r.id, "filename": r.filename, "parsed": r.parsed_json, "ats": r.ats_breakdown,
            "created_at": r.created_at.isoformat()}


@router.post("/jd-match")
def jd_match(body: JDIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    r = _latest(db, user)
    return RS.jd_match(S.ResumeExtraction.model_validate(r.parsed_json), r.raw_text, body.jd_text)


@router.post("/rewrite-bullets")
def rewrite(body: BulletsIn, user: M.User = Depends(require_consent)):
    return RS.rewrite_bullets(body.bullets, body.target_role or user.target_role)


@router.post("/cover-letter")
def cover_letter(body: CoverLetterIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    r = _latest(db, user)
    parsed = S.ResumeExtraction.model_validate(r.parsed_json)
    if not parsed.name:
        parsed.name = user.name
    return RS.cover_letter(parsed, body.jd_text, body.company, body.role, body.tone)
