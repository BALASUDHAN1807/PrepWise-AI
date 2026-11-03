"""Signup/login, profile, consent, data export and "delete all my data"."""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.core.database import get_db
from backend.core.security import create_access_token, get_current_user, hash_password, verify_password
from backend.models import db_models as M
from backend.models.schemas import ConsentIn, DeleteIn, LoginIn, ProfileIn, SignupIn

router = APIRouter(prefix="/auth", tags=["auth"])


def user_public(u: M.User) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name, "role": u.role, "target_role": u.target_role,
            "company_type": u.company_type, "experience_level": u.experience_level,
            "preferred_language": u.preferred_language, "consent_given": u.consent_given}


@router.post("/signup", status_code=201)
def signup(body: SignupIn, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(M.User).where(M.User.email == email)):
        raise HTTPException(409, "An account with this email already exists.")
    role = "admin" if email in get_settings().admin_email_set else "user"
    u = M.User(email=email, name=body.name.strip(), password_hash=hash_password(body.password), role=role,
               consent_given=body.consent, consent_at=datetime.now(UTC) if body.consent else None)
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"token": create_access_token(u.id, u.role), "user": user_public(u)}


@router.post("/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    u = db.scalar(select(M.User).where(M.User.email == body.email.lower()))
    # Same message for unknown email and wrong password (no account enumeration).
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Incorrect email or password.")
    return {"token": create_access_token(u.id, u.role), "user": user_public(u)}


@router.get("/me")
def me(user: M.User = Depends(get_current_user)):
    return user_public(user)


@router.put("/profile")
def update_profile(body: ProfileIn, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(user, k, v.strip() if isinstance(v, str) else v)
    db.commit()
    return user_public(user)


@router.post("/consent")
def set_consent(body: ConsentIn, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.consent_given = body.consent
    user.consent_at = datetime.now(UTC) if body.consent else None
    db.commit()
    return user_public(user)


@router.get("/export")
def export_my_data(user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Right of access: everything stored about the user as JSON (DPDP Act, section 11)."""
    def rows(model, col):
        return [{c.name: (v.isoformat() if isinstance(v := getattr(r, c.name), datetime) else v)
                 for c in model.__table__.columns} for r in db.scalars(select(model).where(col == user.id))]

    return {"profile": user_public(user), "resumes": rows(M.Resume, M.Resume.user_id),
            "sessions": rows(M.InterviewSession, M.InterviewSession.user_id), "answers": rows(M.Answer, M.Answer.user_id),
            "feedback": rows(M.Feedback, M.Feedback.user_id), "roadmaps": rows(M.Roadmap, M.Roadmap.user_id)}


@router.delete("/me")
def delete_my_data(body: DeleteIn, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """One-click erasure: removes the account and (via ON DELETE CASCADE) every row linked to it."""
    if not body.confirm:
        raise HTTPException(400, "Send {\"confirm\": true} to permanently delete your account and all data.")
    db.query(M.PromptVersion).filter(M.PromptVersion.created_by == user.id).update({"created_by": None})
    db.delete(user)
    db.commit()
    return {"deleted": True}
