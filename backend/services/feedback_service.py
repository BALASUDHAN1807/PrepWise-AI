"""Post-session feedback prompt chain, multilingual output and dashboard analytics.

The chain is a FIXED sequence of prompts (no agent decides what to do next):

    summary -> skill-gap mapping -> per-question feedback + model answers
            -> improvement plan -> (optional) translation

Every step is validated JSON with a deterministic fallback. Skill gaps are validated against
``data/skills_taxonomy.json``; unknown skill IDs from the LLM are discarded.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.llm.client import LLMClient
from backend.llm.runner import run_prompt
from backend.models import db_models as M
from backend.models import llm_schemas as S
from backend.services import heuristics as H
from backend.services import rag_service as R

LANG_NAMES = {"en": "English", "ta": "Tamil", "hi": "Hindi"}
DISCLAIMER = ("Scores are guidance only. They come from automated rubrics, test cases and a small classifier, "
              "can be wrong or biased, and are not a hiring decision. Use them to practise, not to judge a person.")
TOPIC_TO_SKILL = {"arrays": "arrays_strings", "strings": "arrays_strings", "linked_lists": "linked_lists",
                  "trees": "trees", "graphs": "graphs", "dp": "dynamic_programming", "sql": "sql_databases"}
CRITERION_TO_SKILL = {"clarity": "communication", "structure": "star_storytelling", "relevance": "communication",
                      "confidence": "confidence_presence", "grammar_vocabulary": "english_fluency",
                      "depth": "depth_of_answer"}


# ------------------------------------------------------------ session data --
def session_results(db: Session, session_id: int) -> list[dict]:
    """Condensed per-question results used as the input of every chain step."""
    answers = db.scalars(select(M.Answer).where(M.Answer.session_id == session_id).order_by(M.Answer.id)).all()
    out = []
    for a in answers:
        if a.score is None:
            continue
        if a.question_type == "dsa":
            t = a.test_results or {}
            rev = (a.evaluation or {}).get("review", {})
            out.append({"id": a.id, "type": "dsa", "title": a.question_json.get("title", ""),
                        "question": a.question_json.get("title", ""), "topic": a.topic, "difficulty": a.difficulty,
                        "score": a.score, "passed": t.get("passed", 0), "total": t.get("total", 0),
                        "time_complexity": rev.get("time_complexity", "?"),
                        "expected_time": a.question_json.get("expected_time_complexity", "?"),
                        "hints_used": a.hints_used, "missed_edge_cases": rev.get("missed_edge_cases", []),
                        "breakdown": a.score_breakdown})
        else:
            rub = (a.evaluation or {}).get("rubric", {})
            scores = {k: rub[k]["score"] for k in S.RubricEvaluation.RUBRIC_KEYS if k in rub and isinstance(rub[k], dict)}
            out.append({"id": a.id, "type": "soft", "question": a.question_json.get("question", ""),
                        "title": a.question_json.get("question", "")[:70], "topic": a.topic, "score": a.score,
                        "rubric": scores, "answer": a.answer_text or "",
                        "classifier": (a.classifier or {}).get("label"),
                        "followup_depth": a.followup_depth, "breakdown": a.score_breakdown})
    return out


def deterministic_skill_gaps(results: list[dict]) -> dict[str, dict]:
    """Rule-based gaps: weak DSA topics (< 60) and low rubric criteria (<= 5)."""
    tax = R.load_taxonomy()
    gaps: dict[str, dict] = {}

    def add(sid: str, reason: str, severity: str):
        if sid in tax and sid not in gaps:
            gaps[sid] = {"skill_id": sid, "reason": reason, "severity": severity}

    for r in results:
        if r["type"] == "dsa" and r["score"] < 60:
            add(TOPIC_TO_SKILL.get(r["topic"], "problem_solving"), f"Scored {r['score']:.0f}/100 on '{r['title']}'",
                "high" if r["score"] < 40 else "medium")
            if r["total"] and r["passed"] < r["total"]:
                add("problem_solving", f"Only {r['passed']}/{r['total']} tests passed", "medium")
            if r.get("missed_edge_cases"):
                add("problem_solving", "Missed edge cases: " + "; ".join(r["missed_edge_cases"][:2]), "medium")
        if r["type"] == "soft":
            for crit, v in r.get("rubric", {}).items():
                if v <= 5:
                    add(CRITERION_TO_SKILL[crit], f"{crit.replace('_', ' ').title()} scored {v}/10", "high" if v <= 3 else "medium")
    return gaps


def _translate(items: list[str], language: str, client: LLMClient | None) -> tuple[list[str], bool]:
    """Translate strings into ta/hi in ONE batch. Returns (items, translated_ok)."""
    if language == "en" or not items:
        return items, True
    res = run_prompt("translate", {"language_name": LANG_NAMES[language], "count": len(items),
                                   "items": items, "_ctx": {"items": items}},
                     fallback=lambda: S.TranslationBatch(items=list(items)), client=client)
    ok = (not res.used_fallback) and len(res.data.items) == len(items) and res.provider != "mock"
    return (res.data.items if len(res.data.items) == len(items) else items), ok


def _collect_strings(content: dict) -> list[tuple[tuple, str]]:
    """Locate every translatable string in the feedback JSON as (path, text)."""
    paths: list[tuple[tuple, str]] = []
    paths.append((("summary",), content["summary"]))
    for key in ("strengths", "weaknesses", "suggestions"):
        for i, s in enumerate(content[key]):
            paths.append(((key, i), s))
    for i, g in enumerate(content["skill_gaps"]):
        paths.append((("skill_gaps", i, "reason"), g["reason"]))
    for i, q in enumerate(content["per_question"]):
        paths.append((("per_question", i, "feedback"), q["feedback"]))
        if q.get("model_answer"):
            paths.append((("per_question", i, "model_answer"), q["model_answer"]))
    return paths


def _set_path(obj, path, value):
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def generate_feedback(db: Session, user: M.User, session: M.InterviewSession, language: str | None = None,
                      client: LLMClient | None = None) -> M.Feedback:
    """Run the feedback chain for a finished session and store the result."""
    import copy

    language = language or user.preferred_language or "en"
    results = session_results(db, session.id)
    if not results:
        raise ValueError("This session has no scored answers yet.")
    tax = R.load_taxonomy()
    overall = round(mean(r["score"] for r in results), 1)
    session.overall_score = overall

    # Step 1: summary
    s1 = run_prompt("feedback_summary", {"kind": session.kind, "overall_score": overall, "results": results,
                                         "_ctx": {"kind": session.kind, "overall": overall, "results": results}},
                    fallback=lambda: H.heuristic_summary(session.kind, overall, results), client=client)
    # Step 2: skill gaps = deterministic rules + LLM mapping (validated against the taxonomy)
    gaps = deterministic_skill_gaps(results)
    low_areas = [f"{r['title']} ({r['score']:.0f})" for r in results if r["score"] < 65]
    s2 = run_prompt("skill_gap_mapping", {"weaknesses": s1.data.weaknesses, "low_areas": low_areas,
                                          "taxonomy": "\n".join(f"{k}: {v['name']}" for k, v in tax.items()),
                                          "_ctx": {"low_areas": s1.data.weaknesses + low_areas, "taxonomy": tax}},
                    fallback=lambda: S.SkillGapMapping(skill_gaps=[]), client=client)
    rejected = []
    for g in s2.data.skill_gaps:
        if g.skill_id in tax:
            gaps.setdefault(g.skill_id, g.model_dump())
        else:
            rejected.append(g.skill_id)
    skill_gaps = [{**g, "name": tax[g["skill_id"]]["name"]} for g in gaps.values()][:8]
    # Step 3: per-question feedback and model answers
    items = [{"type": r["type"], "question": r["question"], "score": r["score"], "rubric": r.get("rubric"),
              "passed": r.get("passed"), "total": r.get("total"), "time_complexity": r.get("time_complexity"),
              "expected_time": r.get("expected_time")} for r in results]
    resume = db.scalars(select(M.Resume).where(M.Resume.user_id == user.id).order_by(M.Resume.id.desc())).first()
    background = (", ".join((resume.parsed_json or {}).get("skills", [])[:8]) if resume else "") or "a final-year CS student"
    s3 = run_prompt("question_feedback", {"background": background, "items": items, "_ctx": {"items": items}},
                    fallback=lambda: H.heuristic_question_feedback(items), client=client)
    per_q = []
    for r, fb in zip(results, list(s3.data.items) + [None] * len(results)):
        base = fb or H.heuristic_question_feedback([items[results.index(r)]]).items[0]
        per_q.append({"answer_id": r["id"], "type": r["type"], "question": r["question"], "score": r["score"],
                      "feedback": base.feedback, "model_answer": base.model_answer if r["type"] == "soft" else ""})
    # Step 4: improvement plan
    gap_names = [g["name"] for g in skill_gaps]
    s4 = run_prompt("improvement_plan", {"weaknesses": s1.data.weaknesses, "skill_gaps": gap_names,
                                         "target_role": user.target_role,
                                         "_ctx": {"weaknesses": s1.data.weaknesses, "gaps": gap_names}},
                    fallback=lambda: H.heuristic_improvement_plan(s1.data.weaknesses, gap_names), client=client)

    content_en = {
        "summary": s1.data.summary, "strengths": s1.data.strengths, "weaknesses": s1.data.weaknesses,
        "skill_gaps": skill_gaps, "per_question": per_q, "suggestions": s4.data.suggestions,
        "overall_score": overall,
        "explanations": [{"question": r["question"], "type": r["type"], "score": r["score"],
                          "breakdown": r.get("breakdown"), "rubric": r.get("rubric")} for r in results],
        "disclaimer": DISCLAIMER,
        "meta": {"chain": ["feedback_summary", "skill_gap_mapping", "question_feedback", "improvement_plan"],
                 "llm": [x.info() for x in (s1, s2, s3, s4)], "rejected_skill_ids": rejected},
    }
    content = copy.deepcopy(content_en)
    if language != "en":
        paths = _collect_strings(content)
        translated, ok = _translate([t for _, t in paths], language, client)
        for (path, _), new in zip(paths, translated):
            _set_path(content, path, new)
        content["meta"]["translation"] = {"language": language, "ok": ok}
        if not ok:
            content["meta"]["translation"]["note"] = "Translation unavailable with this provider; showing English."
    content["language"] = language
    fb = M.Feedback(session_id=session.id, user_id=user.id, language=language, content=content, content_en=content_en)
    db.add(fb)
    session.status = "completed"
    db.commit()
    db.refresh(fb)
    return fb


# ---------------------------------------------------------------- dashboard --
def readiness_weights() -> dict[str, float]:
    s = get_settings()
    return {"resume": s.readiness_weight_resume, "dsa": s.readiness_weight_dsa, "soft": s.readiness_weight_soft}


def compute_readiness(resume: float | None, dsa: float | None, soft: float | None,
                      weights: dict[str, float] | None = None) -> dict:
    """Interview Readiness Score = weighted mean of the available components (weights renormalised)."""
    w = weights or readiness_weights()
    vals = {"resume": resume, "dsa": dsa, "soft": soft}
    avail = {k: v for k, v in vals.items() if v is not None}
    if not avail:
        return {"score": None, "components": vals, "weights": w, "note": "No data yet."}
    tot_w = sum(w[k] for k in avail)
    score = sum(w[k] * v for k, v in avail.items()) / tot_w
    return {"score": round(score, 1), "components": vals,
            "weights": {k: round(w[k] / tot_w, 3) for k in avail},
            "note": "" if len(avail) == 3 else "Missing components are excluded and weights renormalised."}


def dashboard(db: Session, user: M.User) -> dict:
    """Everything the Dashboard page plots: trends, radar, before/after and readiness."""
    resume = db.scalars(select(M.Resume).where(M.Resume.user_id == user.id).order_by(M.Resume.id.desc())).first()
    answers = db.scalars(select(M.Answer).where(M.Answer.user_id == user.id, M.Answer.score.is_not(None))
                         .order_by(M.Answer.id)).all()
    dsa = [a for a in answers if a.question_type == "dsa"]
    soft = [a for a in answers if a.question_type == "soft"]
    recent = lambda xs: round(mean(a.score for a in xs[-10:]), 1) if xs else None  # noqa: E731
    readiness = compute_readiness(resume.ats_score if resume else None, recent(dsa), recent(soft))

    sessions = db.scalars(select(M.InterviewSession).where(M.InterviewSession.user_id == user.id,
                                                           M.InterviewSession.overall_score.is_not(None))
                          .order_by(M.InterviewSession.id)).all()
    trend = [{"session_id": s.id, "kind": s.kind, "score": s.overall_score,
              "date": s.started_at.strftime("%Y-%m-%d %H:%M")} for s in sessions]

    by_topic: dict[str, list[float]] = defaultdict(list)
    for a in dsa:
        by_topic[a.topic].append(a.score)
    crit: dict[str, list[float]] = defaultdict(list)
    for a in soft:
        for k, v in ((a.evaluation or {}).get("rubric") or {}).items():
            if isinstance(v, dict) and "score" in v and k in S.RubricEvaluation.RUBRIC_KEYS:
                crit[k].append(v["score"] * 10)
    radar = {f"DSA: {k}": round(mean(v), 1) for k, v in by_topic.items()}
    radar.update({f"Soft: {k.replace('_', ' ')}": round(mean(v), 1) for k, v in crit.items()})
    if resume:
        radar["Resume (ATS)"] = resume.ats_score

    def before_after(kind: str) -> dict | None:
        xs = [s.overall_score for s in sessions if s.kind in (kind, "mock")]
        if len(xs) < 2:
            return None
        k = max(1, min(3, len(xs) // 2))
        return {"before": round(mean(xs[:k]), 1), "after": round(mean(xs[-k:]), 1), "sessions": len(xs)}

    return {"readiness": readiness, "trend": trend, "radar": radar,
            "before_after": {"dsa": before_after("dsa"), "softskills": before_after("softskills")},
            "counts": {"dsa_answers": len(dsa), "soft_answers": len(soft), "sessions": len(sessions)},
            "resume_ats": resume.ats_score if resume else None, "disclaimer": DISCLAIMER}


def latest_skill_gap_ids(db: Session, user_id: int, limit_sessions: int = 3) -> list[str]:
    """Skill-gap IDs from the most recent feedback rows, most frequent / most severe first."""
    rows = db.scalars(select(M.Feedback).where(M.Feedback.user_id == user_id).order_by(M.Feedback.id.desc())
                      .limit(limit_sessions)).all()
    weight = {"high": 3, "medium": 2, "low": 1}
    score: dict[str, int] = defaultdict(int)
    for fb in rows:
        for g in (fb.content_en or {}).get("skill_gaps", []):
            score[g["skill_id"]] += weight.get(g.get("severity", "medium"), 2)
    return [k for k, _ in sorted(score.items(), key=lambda kv: -kv[1])]
