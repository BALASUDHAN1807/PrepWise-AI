"""Prompt Playground (Prompt IDE): browse versioned templates, edit, run, save and compare versions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from jinja2 import TemplateError
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core import ratelimit
from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.llm.client import LLMClient, LLMError, extract_json, schema_hint
from backend.llm.prompt_loader import _env, list_prompts, load_prompt
from backend.models import db_models as M
from backend.models.llm_schemas import SCHEMA_REGISTRY
from backend.models.schemas import PromptRunIn, PromptVersionIn

router = APIRouter(prefix="/prompts", tags=["prompts"])

# Sample inputs so every template can be tried immediately in the Playground.
_Q = "Tell me about a time you had a disagreement with a teammate. How did you resolve it?"
_A = ("In our database project my teammate wanted MongoDB and I preferred MySQL. I built a small benchmark, "
      "we reviewed the numbers together, chose MySQL and delivered four days early. I learned to move debates to data.")
_RES = {"name": "Asha K", "skills": ["Python", "SQL", "React"], "projects": [{"name": "Campus Queue App", "tech": ["React", "Node.js"]}]}
SAMPLES: dict[str, dict] = {
    "resume_extraction": {"resume_text": "Asha K\nasha@example.com\nEducation\nB.Tech CSE, SRM Institute, 2028, CGPA 8.9\nSkills\nPython, SQL, React\nProjects\nCampus Queue App - built a React and Node.js queue manager used by 200 students",
                          "orgs": ["SRM Institute"], "dates": ["2028"], "taxonomy_skills": ["Python", "SQL", "React"]},
    "resume_suggestions": {"target_role": "Software Engineer", "ats_breakdown": {"keywords": {"score": 18, "max": 30}, "impact": {"score": 6, "max": 25}},
                           "parsed_resume": _RES},
    "jd_match": {"jd_text": "We need a backend engineer with Python, SQL, Docker and REST API experience.", "resume_skills": ["Python", "SQL", "React"],
                 "resume_projects": ["Campus Queue App"], "matched": ["Python", "SQL"], "missing": ["Docker", "REST APIs"], "match_pct": 50},
    "bullet_rewrite": {"target_role": "Software Engineer", "bullets": ["Worked on a website for the college fest", "Helped the team fix bugs"]},
    "cover_letter": {"name": "Asha K", "company": "Acme Labs", "role": "Backend Intern", "tone": "professional", "parsed_resume": _RES,
                     "jd_text": "Acme Labs is hiring a backend intern to build Python and SQL services."},
    "dsa_question_generation": {"topic": "arrays", "difficulty": "easy", "role": "Software Engineer", "company_style": "product",
                                "kind": "code", "skills": ["Python"], "weak_areas": ["graphs"], "avoid_titles": [], "examples": "(none)"},
    "code_review": {"title": "Budget Pair", "statement": "Find two items that sum to the budget.", "expected_time": "O(n)", "expected_space": "O(n)",
                    "language": "python", "passed": 5, "total": 7, "failures": ["case 5: wrong_answer"],
                    "code": "def solve(a, t):\n    for i in range(len(a)):\n        for j in range(i+1, len(a)):\n            if a[i]+a[j]==t:\n                return i+1, j+1"},
    "soft_question_generation": {"company_type": "product", "experience_level": "fresher", "target_role": "Software Engineer", "count": 3,
                                 "categories": ["behavioral", "hr", "resume"], "resume_summary": "Skills: Python, SQL. Projects: Campus Queue App.", "avoid": []},
    "followup_question": {"question": _Q, "answer": "We had a disagreement and then resolved it by talking.", "depth": 0, "max_depth": 2},
    "rubric_evaluation": {"question": _Q, "answer": _A},
    "feedback_summary": {"kind": "softskills", "overall_score": 62, "results": [{"question": _Q, "score": 62, "rubric": {"depth": 4, "structure": 6}}]},
    "skill_gap_mapping": {"weaknesses": ["Answers lack measurable results"], "low_areas": ["depth (4/10)"],
                          "taxonomy": "communication: Communication & Clarity\nstar_storytelling: Structured Answers (STAR)\ndepth_of_answer: Depth & Specifics"},
    "question_feedback": {"background": "Python, SQL", "items": [{"type": "soft", "question": _Q, "score": 62, "rubric": {"depth": 4}}]},
    "improvement_plan": {"weaknesses": ["No measurable results in answers"], "skill_gaps": ["Depth & Specifics"], "target_role": "Software Engineer"},
    "translate": {"language_name": "Hindi", "count": 2, "items": ["Great structure.", "Add a measurable result."]},
    "roadmap_generation": {"target_role": "Backend Engineer", "timeline_months": 3, "total_weeks": 12, "hours_per_week": 8, "skills": ["Python", "SQL"],
                           "skill_gaps": ["Graphs", "System Design"], "plan_weeks": 4,
                           "courses": "coursera-dsa-ucsd | Data Structures and Algorithms Specialization | Coursera | intermediate | Graphs"},
}


def _template_view(t) -> dict:
    return {"name": t.name, "version": t.version, "description": t.description, "temperature": t.temperature,
            "output_schema": t.output_schema, "system": t.system, "user": t.user, "source": "file"}


@router.get("")
def list_all(user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    out = []
    for name, versions in sorted(list_prompts().items()):
        t = load_prompt(name)
        saved = db.scalars(select(M.PromptVersion).where(M.PromptVersion.name == name).order_by(M.PromptVersion.id)).all()
        out.append({"name": name, "file_versions": versions, "description": t.description,
                    "saved_versions": [{"id": s.id, "label": s.version_label, "created_at": s.created_at.isoformat()} for s in saved]})
    return {"prompts": out, "schemas": sorted(SCHEMA_REGISTRY)}


@router.get("/saved/{version_id}")
def saved_version(version_id: int, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    s = db.get(M.PromptVersion, version_id)
    if not s:
        raise HTTPException(404, "Saved version not found.")
    return {"id": s.id, "name": s.name, "version": s.version_label, "system": s.system, "user": s.user_template,
            "temperature": s.temperature, "notes": s.notes, "source": "saved", "created_at": s.created_at.isoformat()}


@router.get("/{name}")
def get_prompt(name: str, version: int | None = None, user: M.User = Depends(get_current_user)):
    try:
        t = load_prompt(name, version)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {**_template_view(t), "sample_variables": SAMPLES.get(name, {})}


@router.post("/versions", status_code=201)
def save_version(body: PromptVersionIn, user: M.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if body.name not in list_prompts():
        raise HTTPException(422, f"Unknown prompt '{body.name}'.")
    row = M.PromptVersion(name=body.name, version_label=body.version_label, system=body.system,
                          user_template=body.user_template, temperature=body.temperature, notes=body.notes,
                          created_by=user.id)
    db.add(row)
    db.commit()
    return {"id": row.id, "name": row.name, "label": row.version_label}


@router.post("/run")
def run(body: PromptRunIn, user: M.User = Depends(get_current_user)):
    """Render the (edited) template, call the chosen model and report JSON validity."""
    ratelimit.check(f"playground:{user.id}", max_calls=20, window_sec=600)
    try:
        system = _env.from_string(body.system).render(**body.variables)
        user_msg = _env.from_string(body.user_template).render(**body.variables)
    except TemplateError as exc:
        raise HTTPException(422, f"Template error: {exc}") from exc
    schema_name = body.schema_name
    if not schema_name and body.prompt_name:
        try:
            schema_name = load_prompt(body.prompt_name).output_schema
        except KeyError:
            schema_name = None
    schema = SCHEMA_REGISTRY.get(schema_name or "")
    if schema and body.append_schema:
        system += ("\n\nReturn ONLY a JSON object that matches this JSON schema (no markdown, no commentary):\n"
                   + schema_hint(schema))
    try:
        client = LLMClient(provider=body.provider, model=body.model or None)
        res = client.complete(system, user_msg, body.temperature, json_mode=bool(schema),
                              task=body.prompt_name, variables=body.variables)
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    errors, json_valid, schema_valid, parsed = [], False, None, None
    try:
        parsed = extract_json(res.text)
        json_valid = True
        if schema:
            try:
                schema.model_validate(parsed)
                schema_valid = True
            except ValidationError as exc:
                schema_valid = False
                errors = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:8]]
    except ValueError as exc:
        errors = [str(exc)]
    return {"output": res.text, "parsed": parsed, "json_valid": json_valid, "schema": schema_name,
            "schema_valid": schema_valid, "errors": errors, "provider": res.provider, "model": res.model,
            "latency_ms": round(res.latency_ms, 1), "temperature": body.temperature,
            "rendered_user_prompt": user_msg[:4000]}
