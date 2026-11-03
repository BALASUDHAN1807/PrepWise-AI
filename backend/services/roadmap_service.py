"""Career roadmap generation: RAG over the course catalog + strict course validation.

* Skill gaps come from the resume, the last feedback reports and the target role.
* ``rag_service.retrieve_courses`` returns candidate courses ONLY from ``data/courses.json``.
* The LLM may recommend courses ONLY by ``course_id`` from that candidate list;
  :func:`validate_roadmap_courses` drops anything else (reported in ``rejected_course_ids``)
  and fills in title / platform / url from the catalog, never from the model.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.llm.client import LLMClient
from backend.llm.runner import run_prompt
from backend.models import db_models as M
from backend.models import llm_schemas as S
from backend.services import feedback_service as F
from backend.services import heuristics as H
from backend.services import rag_service as R

DEFAULT_GAPS = ["arrays_strings", "trees", "graphs", "dynamic_programming", "sql_databases", "communication"]
ROLE_GAPS = {
    "ml": ["python", "machine_learning", "deep_learning", "generative_ai_llms", "arrays_strings", "communication"],
    "data": ["python", "sql_databases", "machine_learning", "arrays_strings", "communication"],
    "frontend": ["web_development", "react_frontend", "arrays_strings", "git_version_control", "communication"],
    "backend": ["sql_databases", "web_development", "system_design", "computer_networks", "trees", "graphs"],
    "devops": ["computer_networks", "git_version_control", "cybersecurity", "python", "communication"],
}


def role_default_gaps(role: str) -> list[str]:
    t = role.lower()
    for key, hints in [("ml", ["machine learning", "ml", "ai", "nlp"]), ("data", ["data"]),
                       ("frontend", ["front"]), ("backend", ["back", "full"]), ("devops", ["devops", "cloud", "sre"])]:
        if any(h in t.split() or h in t for h in hints):
            return ROLE_GAPS[key]
    return DEFAULT_GAPS


def validate_roadmap_courses(output: S.RoadmapOutput, retrieved_ids: set[str], catalog: dict[str, dict]
                             ) -> tuple[list[dict], list[str]]:
    """Keep only recommendations whose id exists in the catalog AND was retrieved.

    Returns (valid recommendations enriched from the catalog, rejected ids).
    """
    valid, rejected, seen = [], [], set()
    for rec in output.recommended_courses:
        cid = rec.course_id.strip()
        if cid in catalog and cid in retrieved_ids:
            if cid in seen:
                continue
            seen.add(cid)
            c = catalog[cid]
            valid.append({"course_id": cid, "title": c["title"], "platform": c["platform"], "level": c["level"],
                          "url": c["url"], "provider": c.get("provider", ""), "skill": rec.skill, "why": rec.why})
        else:
            rejected.append(cid)
    return valid, rejected


def build_roadmap(target_role: str, timeline_months: int, hours_per_week: int, skills: list[str],
                  gap_ids: list[str], client: LLMClient | None = None) -> dict:
    """Pure generation step (no DB) - also used by tests and scripts."""
    tax, catalog = R.load_taxonomy(), {c["id"]: c for c in R.load_courses()}
    gap_ids = [g for g in gap_ids if g in tax] or role_default_gaps(target_role)
    candidates = R.retrieve_courses(gap_ids[:6], k_per_skill=2)
    retrieved_ids = {c["id"] for c in candidates}
    total_weeks = timeline_months * 4
    plan_weeks = min(total_weeks, 12)
    course_lines = "\n".join(f"{c['id']} | {c['title']} | {c['platform']} | {c['level']} | "
                             f"{', '.join(tax[s]['name'] for s in c['skills'] if s in tax)}" for c in candidates)
    gap_names = [tax[g]["name"] for g in gap_ids]
    ctx = dict(target_role=target_role, timeline_months=timeline_months, hours_per_week=hours_per_week,
               skills=skills, skill_gaps=gap_ids, courses=candidates, plan_weeks=plan_weeks)
    res = run_prompt(
        "roadmap_generation",
        {"target_role": target_role, "timeline_months": timeline_months, "total_weeks": total_weeks,
         "hours_per_week": hours_per_week, "skills": skills[:25], "skill_gaps": gap_names, "courses": course_lines,
         "plan_weeks": plan_weeks, "_ctx": ctx},
        fallback=lambda: H.heuristic_roadmap(**ctx), client=client)
    out = res.data
    valid, rejected = validate_roadmap_courses(out, retrieved_ids, catalog)
    if not valid:  # LLM gave nothing usable -> top retrieved courses
        valid = [{"course_id": c["id"], "title": c["title"], "platform": c["platform"], "level": c["level"],
                  "url": c["url"], "provider": c.get("provider", ""), "skill": c["matched_skills"][0],
                  "why": "Retrieved for your skill gap."} for c in candidates[:5]]
    content = out.model_dump()
    content["recommended_courses"] = valid
    content["meta"] = {"rejected_course_ids": rejected, "retrieved_course_ids": sorted(retrieved_ids),
                       "skill_gap_names": gap_names, "llm": res.info(), "embedding": R.embedding_info()}
    return {"content": content, "gap_ids": gap_ids}


def progress_keys(content: dict) -> list[str]:
    keys = []
    for w in content.get("weekly_plan", []):
        keys += [f"w{w['week']}-t{i}" for i in range(len(w["tasks"]))]
    keys += [f"m{i}" for i in range(len(content.get("milestones", [])))]
    keys += [f"c-{c['course_id']}" for c in content.get("recommended_courses", [])]
    return keys


def progress_summary(db: Session, roadmap: M.Roadmap) -> dict:
    keys = progress_keys(roadmap.content)
    done = {p.item_key for p in db.scalars(select(M.RoadmapProgress).where(
        M.RoadmapProgress.roadmap_id == roadmap.id, M.RoadmapProgress.done.is_(True)))}
    done &= set(keys)
    return {"done": sorted(done), "total": len(keys), "completed": len(done),
            "percent": round(100 * len(done) / len(keys), 1) if keys else 0.0}


def create_roadmap(db: Session, user: M.User, target_role: str, timeline_months: int, hours_per_week: int,
                   client: LLMClient | None = None) -> M.Roadmap:
    if timeline_months not in (3, 6, 12):
        raise ValueError("timeline_months must be 3, 6 or 12")
    if not 1 <= hours_per_week <= 80:
        raise ValueError("hours_per_week must be between 1 and 80")
    resume = db.scalars(select(M.Resume).where(M.Resume.user_id == user.id).order_by(M.Resume.id.desc())).first()
    skills = list((resume.parsed_json or {}).get("skills", [])) if resume else []
    gap_ids = F.latest_skill_gap_ids(db, user.id)
    gen = build_roadmap(target_role, timeline_months, hours_per_week, skills, gap_ids, client)
    for old in db.scalars(select(M.Roadmap).where(M.Roadmap.user_id == user.id, M.Roadmap.is_active.is_(True))):
        old.is_active = False
    rm = M.Roadmap(user_id=user.id, target_role=target_role, timeline_months=timeline_months,
                   hours_per_week=hours_per_week, skill_gaps=gen["gap_ids"], content=gen["content"], is_active=True)
    db.add(rm)
    db.commit()
    db.refresh(rm)
    return rm
