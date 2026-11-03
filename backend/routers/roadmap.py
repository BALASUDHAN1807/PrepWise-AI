"""Career roadmap: generate, view, track progress, regenerate from the latest skill gaps."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import get_current_user, require_consent
from backend.models import db_models as M
from backend.models.schemas import ProgressIn, RoadmapIn
from backend.services import roadmap_service as RM

router = APIRouter(prefix="/roadmap", tags=["roadmap"])


def _view(db: Session, rm: M.Roadmap) -> dict:
    return {"id": rm.id, "target_role": rm.target_role, "timeline_months": rm.timeline_months,
            "hours_per_week": rm.hours_per_week, "skill_gaps": rm.skill_gaps, "content": rm.content,
            "is_active": rm.is_active, "created_at": rm.created_at.isoformat(), "progress": RM.progress_summary(db, rm)}


def _own(db: Session, user: M.User, rid: int) -> M.Roadmap:
    rm = db.get(M.Roadmap, rid)
    if not rm or rm.user_id != user.id:
        raise HTTPException(404, "Roadmap not found.")
    return rm


@router.post("/generate")
def generate(body: RoadmapIn, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    try:
        rm = RM.create_roadmap(db, user, body.target_role or user.target_role, body.timeline_months, body.hours_per_week)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _view(db, rm)


@router.get("/active")
def active(user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    rm = db.scalars(select(M.Roadmap).where(M.Roadmap.user_id == user.id, M.Roadmap.is_active.is_(True))
                    .order_by(M.Roadmap.id.desc())).first()
    if not rm:
        raise HTTPException(404, "You have no roadmap yet.")
    return _view(db, rm)


@router.post("/{roadmap_id}/progress")
def progress(roadmap_id: int, body: ProgressIn, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    rm = _own(db, user, roadmap_id)
    if body.item_key not in set(RM.progress_keys(rm.content)):
        raise HTTPException(422, "Unknown roadmap item.")
    row = db.scalars(select(M.RoadmapProgress).where(M.RoadmapProgress.roadmap_id == rm.id,
                                                     M.RoadmapProgress.item_key == body.item_key)).first()
    if row:
        row.done = body.done
    else:
        db.add(M.RoadmapProgress(roadmap_id=rm.id, item_key=body.item_key, done=body.done))
    db.commit()
    return RM.progress_summary(db, rm)


@router.post("/{roadmap_id}/regenerate")
def regenerate(roadmap_id: int, user: M.User = Depends(require_consent), db: Session = Depends(get_db)):
    """New roadmap with the same target/timeline but using the LATEST skill gaps from feedback."""
    old = _own(db, user, roadmap_id)
    rm = RM.create_roadmap(db, user, old.target_role, old.timeline_months, old.hours_per_week)
    return _view(db, rm)
