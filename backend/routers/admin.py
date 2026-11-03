"""Admin / enterprise dashboard for a placement cell or HR team.

Privacy by default: only users who gave consent are counted, individuals are shown by a salted
pseudonym (never name/email), and per-student views and CSV exports are withheld for batches
smaller than ``ADMIN_MIN_GROUP`` (k-anonymity style small-cell suppression).
"""
from __future__ import annotations

import csv
import io
from collections import Counter
from statistics import mean

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.core.database import get_db
from backend.core.security import anonymize_id, require_admin
from backend.models import db_models as M
from backend.services import feedback_service as F
from backend.services import rag_service as R

router = APIRouter(prefix="/admin", tags=["admin"])
BINS = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def _bin(x: float) -> str:
    return BINS[min(4, int(x // 20))]


def _batch_rows(db: Session) -> list[dict]:
    users = db.scalars(select(M.User).where(M.User.role == "user", M.User.consent_given.is_(True))).all()
    rows = []
    for u in users:
        resume = db.scalars(select(M.Resume).where(M.Resume.user_id == u.id).order_by(M.Resume.id.desc())).first()
        ans = db.scalars(select(M.Answer).where(M.Answer.user_id == u.id, M.Answer.score.is_not(None))
                         .order_by(M.Answer.id)).all()
        dsa = [a.score for a in ans if a.question_type == "dsa"]
        soft = [a.score for a in ans if a.question_type == "soft"]
        sess = db.scalars(select(M.InterviewSession).where(M.InterviewSession.user_id == u.id,
                                                           M.InterviewSession.overall_score.is_not(None))
                          .order_by(M.InterviewSession.id)).all()
        d10, s10 = (round(mean(dsa[-10:]), 1) if dsa else None), (round(mean(soft[-10:]), 1) if soft else None)
        ready = F.compute_readiness(resume.ats_score if resume else None, d10, s10)["score"]
        gaps = F.latest_skill_gap_ids(db, u.id)
        rows.append({"anon_id": anonymize_id(u.id), "sessions": len(sess), "ats": resume.ats_score if resume else None,
                     "dsa": d10, "soft": s10, "readiness": ready, "gaps": gaps,
                     "first": sess[0].overall_score if sess else None, "last": sess[-1].overall_score if sess else None})
    return rows


@router.get("/overview")
def overview(admin: M.User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = _batch_rows(db)
    k = get_settings().admin_min_group
    tax = R.load_taxonomy()
    active = [r for r in rows if r["readiness"] is not None]
    gap_counts = Counter(g for r in active for g in r["gaps"][:5])
    dist = lambda key: dict(Counter(_bin(r[key]) for r in rows if r[key] is not None))  # noqa: E731
    improvers = sorted((r for r in rows if r["sessions"] >= 2 and r["first"] is not None),
                       key=lambda r: -(r["last"] - r["first"]))[:5]
    suppressed = len(rows) < k
    return {
        "batch": {"consenting_users": len(rows), "with_activity": len(active),
                  "avg_readiness": round(mean(r["readiness"] for r in active), 1) if active else None,
                  "avg_ats": round(mean(r["ats"] for r in rows if r["ats"] is not None), 1)
                  if any(r["ats"] is not None for r in rows) else None},
        "distributions": {"readiness": dist("readiness"), "dsa": dist("dsa"), "soft": dist("soft"), "bins": BINS},
        "common_skill_gaps": [{"skill_id": s, "name": tax.get(s, {}).get("name", s), "students": n,
                               "share_pct": round(100 * n / len(active), 1)} for s, n in gap_counts.most_common(8)] if active else [],
        "top_improvers": [] if suppressed else [
            {"student": r["anon_id"], "first_score": r["first"], "latest_score": r["last"],
             "improvement": round(r["last"] - r["first"], 1), "sessions": r["sessions"]} for r in improvers],
        "privacy": {"anonymized": True, "min_group": k, "per_student_views_suppressed": suppressed,
                    "note": "Only consenting students are counted; names and emails are never shown. "
                            f"Per-student views and CSV export need at least {k} students."},
    }


@router.get("/export.csv")
def export_csv(admin: M.User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = _batch_rows(db)
    k = get_settings().admin_min_group
    if len(rows) < k:
        raise HTTPException(409, f"Export is disabled for batches smaller than {k} consenting students (privacy).")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["student", "sessions", "ats_score", "dsa_avg", "soft_avg", "readiness", "top_skill_gaps"])
    tax = R.load_taxonomy()
    for r in sorted(rows, key=lambda r: r["anon_id"]):
        w.writerow([r["anon_id"], r["sessions"], r["ats"], r["dsa"], r["soft"], r["readiness"],
                    "; ".join(tax.get(g, {}).get("name", g) for g in r["gaps"][:3])])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="prepwise-batch-anonymized.csv"'})
