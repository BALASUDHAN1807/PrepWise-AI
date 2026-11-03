"""Responses for ``LLM_PROVIDER=mock`` (CI, tests and offline demos).

Services pass their structured inputs to the prompt as ``variables["_ctx"]``
(templates ignore it). The mock routes each task to the same deterministic
heuristic the service would use as its fallback, so the whole pipeline runs
without network access while still exercising JSON validation.
"""
from __future__ import annotations

import hashlib
import re

from backend.services import heuristics as H


def _qa_from_text(variables: dict, user: str) -> tuple[str, str]:
    q = variables.get("question") or ""
    a = variables.get("answer") or ""
    if not (q and a):
        mq = re.search(r"Question:\s*(.*)", user)
        ma = re.search(r"Answer:\s*(.*)", user, re.DOTALL)
        q = q or (mq.group(1).strip() if mq else "")
        a = a or (ma.group(1).strip() if ma else "")
    return q, a


def _question_from_bank(ctx: dict) -> dict:
    from backend.services.rag_service import load_question_bank

    bank = load_question_bank()
    topic, diff, kind = ctx.get("topic", ""), ctx.get("difficulty", ""), ctx.get("kind", "code")
    avoid = {t.lower() for t in ctx.get("avoid_titles", [])}
    pool = [q for q in bank if q.get("kind", "code") == kind and q["topic"] == topic and q["difficulty"] == diff]
    pool = pool or [q for q in bank if q.get("kind", "code") == kind and q["topic"] == topic]
    pool = pool or [q for q in bank if q.get("kind", "code") == kind]
    fresh = [q for q in pool if q["title"].lower() not in avoid] or pool
    seed = int(hashlib.md5(f"{topic}{diff}{len(avoid)}".encode()).hexdigest(), 16)
    q = dict(fresh[seed % len(fresh)])
    return {k: q[k] for k in q if k not in {"id", "company_styles", "source"}}


def respond(task: str, variables: dict, system: str, user: str) -> dict:
    """Return a schema-valid dict for ``task``."""
    ctx = variables.get("_ctx", {}) if isinstance(variables, dict) else {}
    if task == "resume_extraction":
        return H.heuristic_resume_extraction(ctx.get("text", ""), ctx.get("taxonomy_skills", [])).model_dump()
    if task == "resume_suggestions":
        return H.heuristic_resume_suggestions(ctx.get("breakdown", {})).model_dump()
    if task == "jd_match":
        return H.heuristic_jd_reasoning(ctx.get("jd_skills", []), ctx.get("matched", []),
                                        ctx.get("missing", []), ctx.get("match_pct", 0.0)).model_dump()
    if task == "bullet_rewrite":
        return {"items": [H.rewrite_bullet(b).model_dump() for b in ctx.get("bullets", variables.get("bullets", []))]}
    if task == "cover_letter":
        return H.heuristic_cover_letter(ctx.get("name", ""), ctx.get("company", ""), ctx.get("role", ""),
                                        ctx.get("skills", []), ctx.get("projects", [])).model_dump()
    if task == "dsa_question_generation":
        return _question_from_bank(ctx)
    if task == "code_review":
        return H.heuristic_code_review(ctx.get("code", ""), ctx.get("language", "python"), ctx.get("passed", 0),
                                       ctx.get("total", 0), ctx.get("expected_time", "O(n)")).model_dump()
    if task == "soft_question_generation":
        return H.heuristic_soft_questions(ctx.get("count", 3), ctx.get("categories", []), ctx.get("target_role", ""),
                                          ctx.get("company_type", ""), ctx.get("projects", []),
                                          ctx.get("skills", []), ctx.get("avoid", [])).model_dump()
    if task == "followup_question":
        q, a = _qa_from_text(variables, user)
        return H.heuristic_followup(q, a, int(variables.get("depth", 0)), int(variables.get("max_depth", 2))).model_dump()
    if task.startswith("rubric_evaluation") or task.startswith("exp_rubric"):
        q, a = _qa_from_text(variables, user)
        ev = H.heuristic_rubric(q, a).model_dump()
        if task == "exp_rubric_cot":
            return {"reasoning": "Rule-based mock: no reasoning trace.", "evaluation": ev}
        return ev
    if task.startswith("exp_feedback"):  # scripts/compare_approaches.py: deliberately naive offline stand-ins
        if task == "exp_feedback_rag":  # behaves like nearest-neighbour copy: reuse the first retrieved example's feedback
            m = re.search(r"Feedback:\s*(.+)", user)
            if m:
                return {"feedback": m.group(1).strip()}
        return {"feedback": "Verdict: average. Strength: You answered the question. Improve depth: add more detail. "
                            "Next step: practise more."}
    if task == "feedback_summary":
        return H.heuristic_summary(ctx.get("kind", ""), ctx.get("overall", 0.0), ctx.get("results", [])).model_dump()
    if task == "skill_gap_mapping":
        return H.heuristic_skill_gaps(ctx.get("low_areas", []), ctx.get("taxonomy", {})).model_dump()
    if task == "question_feedback":
        return H.heuristic_question_feedback(ctx.get("items", [])).model_dump()
    if task == "improvement_plan":
        return H.heuristic_improvement_plan(ctx.get("weaknesses", []), ctx.get("gaps", [])).model_dump()
    if task == "translate":
        # The mock cannot translate; it returns the English text unchanged.
        return {"items": list(ctx.get("items", []))}
    if task == "roadmap_generation":
        return H.heuristic_roadmap(**ctx).model_dump()
    return {"note": "mock provider: no handler for this prompt", "echo": user[:300]}
