"""Deterministic, rule-based stand-ins for every LLM step.

These functions serve two purposes:

1. **Safe fallbacks**: when an LLM response is still invalid after the retries,
   services return these results (flagged ``used_fallback``) instead of failing.
2. **Mock provider**: ``LLM_PROVIDER=mock`` routes every prompt here so CI and
   offline demos run end-to-end without an API key.

They are intentionally simple and transparent. They are NOT a substitute for a
real LLM and the UI labels results produced by them.
"""
from __future__ import annotations

import re
from collections import Counter

from backend.models import llm_schemas as S

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'+#.-]*")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|x|k|ms|s|hrs?|hours?|days?|weeks?|users?|students?|lakh|crore)?",
                       re.IGNORECASE)
STOPWORDS = set(
    """a an the and or but if to of in on at for with from by as is are was were be been being i
    me my we our you your he she they them it its this that these those do did does have has had
    not no so than then there here what which who whom how why when where can could would should
    will just about into over under again further once very also too more most such only own same
    tell describe time give example""".split()
)
HEDGES = ["i think", "maybe", "kind of", "sort of", "i guess", "probably", "i'm not sure",
          "not sure", "i feel like", "perhaps", "somewhat"]
FILLERS_DEFAULT = ["um", "uh", "like", "basically", "actually", "you know", "literally", "so yeah"]
ACTION_VERBS = [
    "built", "led", "designed", "developed", "created", "implemented", "organized", "organised",
    "managed", "improved", "reduced", "increased", "launched", "resolved", "analyzed", "analysed",
    "coordinated", "decided", "proposed", "wrote", "tested", "deployed", "mentored", "negotiated",
    "optimized", "optimised", "automated", "presented", "collaborated", "delivered", "fixed",
    "set up", "took", "initiated", "convinced", "scheduled", "planned", "learned", "learnt",
]
STRONG_VERB_FOR = {
    "worked on": "Developed", "responsible for": "Owned", "helped": "Contributed to",
    "made": "Built", "did": "Executed", "handled": "Managed", "involved in": "Contributed to",
    "participated in": "Contributed to", "was part of": "Collaborated on", "used": "Applied",
}


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def content_words(text: str) -> set[str]:
    return {w.lower() for w in words(text) if w.lower() not in STOPWORDS and len(w) > 2}


def _clamp(v: float, lo: int = 0, hi: int = 10) -> int:
    return int(max(lo, min(hi, round(v))))


# ------------------------------------------------------------- soft skills --
def detect_star(answer: str) -> S.StarDetection:
    """Keyword-based STAR component detection."""
    t = " " + (answer or "").lower() + " "
    situation = any(k in t for k in [" when ", " during ", " in my ", " while ", " at my ", "project",
                                     "internship", "semester", " last year", " once ", "hackathon"])
    task = any(k in t for k in ["my role", "responsible", " task", " goal", "needed to", "had to",
                                "assigned", "objective", "challenge was", "deadline"])
    action = bool(re.search(r"\bi (?:\w+ed|built|led|took|made|wrote|ran|set|began|spoke|met|chose|"
                            r"broke|drew|went|did|got|gave|sat)\b", t)) or any(f" {v} " in t for v in ACTION_VERBS)
    result = bool(NUMBER_RE.search(answer or "")) and any(
        k in t for k in ["result", "outcome", "improv", "reduc", "increas", "deliver", "achiev", "saved",
                         "won", "finished", "completed", "%"]
    ) or any(k in t for k in ["as a result", "the outcome", "in the end", "eventually we", "we delivered",
                               "we won", "i learned", "i learnt"])
    found = [n for n, f in zip("STAR", [situation, task, action, result]) if f]
    comment = f"Detected: {', '.join(found) or 'none'} (keyword heuristic)."
    return S.StarDetection(situation=situation, task=task, action=action, result=result, comment=comment)


def count_fillers(text: str, fillers: list[str] | None = None) -> dict[str, int]:
    t = " " + re.sub(r"[^a-z' ]", " ", (text or "").lower()) + " "
    out = {}
    for f in fillers or FILLERS_DEFAULT:
        n = len(re.findall(rf"(?<![a-z']){re.escape(f)}(?![a-z'])", t))
        if n:
            out[f] = n
    return out


def heuristic_rubric(question: str, answer: str) -> S.RubricEvaluation:
    """Transparent rule-based rubric used as a fallback and by the mock provider."""
    answer = answer or ""
    wc = len(words(answer))
    sentences = [s for s in re.split(r"[.!?]+", answer) if s.strip()]
    avg_len = wc / max(1, len(sentences))
    star = detect_star(answer)
    star_n = sum([star.situation, star.task, star.action, star.result])
    has_num = bool(NUMBER_RE.search(answer))
    low = answer.lower()
    hedges = sum(low.count(h) for h in HEDGES)
    fillers = sum(count_fillers(answer).values())
    reflection = any(k in low for k in ["learned", "learnt", "next time", "in hindsight", "realised",
                                         "realized", "taught me"])
    overlap = content_words(question) & content_words(answer)
    q_words = content_words(question) or {"x"}
    overlap_ratio = len(overlap) / len(q_words)
    first_person = len(re.findall(r"\bi\b", low))

    clarity = 8 - (2 if avg_len > 35 else 0) - min(3, fillers * 0.5)
    structure = 2 + 2 * star_n
    relevance = 3 + overlap_ratio * 8 + (2 if wc > 40 else 0)
    confidence = 7 - hedges + min(2, first_person * 0.3) - min(2, fillers * 0.3)
    grammar = 7 + (1 if answer[:1].isupper() else -1) + (1 if re.search(r"[.!?]\s*$", answer) else -1)
    depth = min(10, wc / 12) * 0.5 + (2 if has_num else 0) + (1.5 if star.result else 0) + (1.5 if reflection else 0)

    if wc < 25:  # one-liners cannot score well on structure/depth/clarity
        structure, depth, clarity = min(structure, 3), min(depth, 2), min(clarity, 5)
        relevance = min(relevance, 4)
        confidence = min(confidence, 5)

    def item(score: float, reason: str) -> S.RubricItem:
        return S.RubricItem(score=_clamp(score), reason=reason)

    return S.RubricEvaluation(
        clarity=item(clarity, f"Average sentence length {avg_len:.0f} words; {fillers} filler word(s)."),
        structure=item(structure, f"{star_n}/4 STAR components detected."),
        relevance=item(relevance, f"{len(overlap)} key term(s) from the question addressed; {wc} words."),
        confidence=item(confidence, f"{hedges} hedge phrase(s), {first_person} first-person statement(s)."),
        grammar_vocabulary=item(grammar, "Checked capitalisation and sentence endings (heuristic)."),
        depth=item(depth, ("Includes" if has_num else "No") + " measurable detail; "
                   + ("reflects on learning." if reflection else "no reflection on learning.")),
        star=star,
        overall_comment="Rule-based estimate (fallback/mock). Use a real LLM provider for nuanced feedback.",
    )


def heuristic_followup(question: str, answer: str, depth: int, max_depth: int = 2) -> S.FollowUp:
    if depth >= max_depth:
        return S.FollowUp(needs_followup=False, reason="Maximum follow-ups reached.")
    star = detect_star(answer)
    wc = len(words(answer))
    if wc < 40 or not star.situation:
        return S.FollowUp(needs_followup=True,
                          question="Can you walk me through one specific situation where this happened, "
                                   "and what exactly you did?",
                          reason="Answer lacks a concrete example.")
    if not star.result:
        return S.FollowUp(needs_followup=True,
                          question="What was the measurable outcome, and what would you do differently next time?",
                          reason="Answer lacks a result or impact.")
    return S.FollowUp(needs_followup=False, reason="Answer is specific and complete.")


SOFT_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "behavioral": [
        "Tell me about a time you had a disagreement with a teammate. How did you resolve it?",
        "Describe a situation where you missed a deadline or nearly did. What did you do?",
        "Give an example of a time you learned a new technology quickly to finish a task.",
        "Tell me about a mistake you made in a project and how you handled it.",
    ],
    "situational": [
        "If your team lead asked you to ship a feature you believed was not ready, what would you do?",
        "Suppose two stakeholders give you conflicting requirements on the same day. How do you proceed?",
        "Imagine production breaks an hour before a demo and you wrote the failing code. What do you do?",
    ],
    "hr": [
        "Why do you want to join a {company_type} company as a {target_role}?",
        "Where do you see yourself in three years?",
        "What is one weakness you are actively working on?",
        "Why should we hire you over other candidates with similar skills?",
    ],
    "resume": [
        "Walk me through your project '{project}'. What was the hardest technical decision you made?",
        "In '{project}', what would you change if you rebuilt it today, and why?",
        "You list {skill} on your resume. Describe a concrete problem you solved with it.",
    ],
}


def heuristic_soft_questions(count: int, categories: list[str], target_role: str, company_type: str,
                             projects: list[str], skills: list[str], avoid: list[str]) -> S.SoftQuestionSet:
    cats = [c for c in categories if c in SOFT_QUESTION_TEMPLATES] or list(SOFT_QUESTION_TEMPLATES)
    project = projects[0] if projects else "your final-year project"
    skill = skills[0] if skills else "Python"
    out: list[S.SoftQuestion] = []
    seen = set(a.strip().lower() for a in avoid)
    idx = {c: 0 for c in cats}
    guard = 0
    while len(out) < count and guard < 50:
        guard += 1
        c = cats[len(out) % len(cats)]
        pool = SOFT_QUESTION_TEMPLATES[c]
        if idx[c] >= len(pool):
            continue
        q = pool[idx[c]].format(company_type=company_type, target_role=target_role, project=project, skill=skill)
        idx[c] += 1
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(S.SoftQuestion(question=q, category=c,
                                  what_good_looks_like="Specific example, clear STAR structure, measurable result, reflection."))
    if not out:
        out.append(S.SoftQuestion(question="Tell me about yourself.", category="hr"))
    return S.SoftQuestionSet(questions=out)


# -------------------------------------------------------------------- DSA --
COMPLEXITY_RANK = [("o(1)", 0), ("o(log n)", 1), ("o(n log n)", 3), ("o(n^2)", 4), ("o(n²)", 4),
                   ("o(n*m)", 4), ("o(n^3)", 5), ("o(2^n)", 6), ("o(n!)", 7), ("o(n)", 2)]


def complexity_rank(expr: str) -> int | None:
    e = (expr or "").lower().replace(" ", "")
    e = e.replace("logn", "log n").replace("nlog n", "n log n").replace("n*log n", "n log n")
    for key, rank in sorted(COMPLEXITY_RANK, key=lambda kv: -len(kv[0])):
        if key.replace(" ", "") in e.replace(" ", ""):
            return rank
    if "v+e" in e or "n+m" in e:
        return 2
    return None


def _loop_depth(code: str, language: str) -> int:
    max_depth = 0
    if language == "python":
        stack: list[int] = []
        for line in code.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            while stack and indent <= stack[-1]:
                stack.pop()
            if re.match(r"\s*(for|while)\b", line):
                stack.append(indent)
                max_depth = max(max_depth, len(stack))
    else:
        depth_stack: list[bool] = []
        loops_open = 0
        pending_loop = False
        for ch_line in code.splitlines():
            if re.search(r"\b(for|while)\s*\(", ch_line):
                pending_loop = True
                max_depth = max(max_depth, loops_open + 1)
            for ch in ch_line:
                if ch == "{":
                    depth_stack.append(pending_loop)
                    if pending_loop:
                        loops_open += 1
                    pending_loop = False
                elif ch == "}" and depth_stack:
                    if depth_stack.pop():
                        loops_open -= 1
    return max_depth


def heuristic_code_review(code: str, language: str, passed: int, total: int,
                          expected_time: str, expected_space: str = "O(n)") -> S.CodeReview:
    """Static analysis estimate of complexity/quality (fallback + mock)."""
    code = code or ""
    depth = _loop_depth(code, language)
    uses_sort = bool(re.search(r"\bsort|sorted\(|Arrays\.sort|Collections\.sort|std::sort", code))
    recursion = False
    m = re.search(r"def\s+(\w+)\s*\(", code)
    if m and len(re.findall(rf"\b{m.group(1)}\s*\(", code)) > 1:
        recursion = True
    guess = {0: "O(n)", 1: "O(n)", 2: "O(n^2)", 3: "O(n^3)"}.get(depth, "O(n^3)")
    if uses_sort and depth <= 1:
        guess = "O(n log n)"
    if recursion and depth == 0:
        guess = "O(n) to O(2^n) depending on memoization"
    exp_rank = complexity_rank(expected_time)
    got_rank = complexity_rank(guess)
    if exp_rank is None or got_rank is None:
        cscore = 6
    else:
        diff = got_rank - exp_rank
        cscore = 10 if diff <= 0 else 7 if diff == 1 else 4 if diff == 2 else 2
    pass_rate = passed / total if total else 0
    approach = _clamp(3 + 7 * pass_rate - (1 if cscore < 7 else 0))
    lines = [ln for ln in code.splitlines() if ln.strip()]
    comments = sum(1 for ln in lines if ln.strip().startswith(("#", "//", "/*", "*")))
    long_lines = sum(1 for ln in lines if len(ln) > 110)
    has_functions = bool(re.search(r"\bdef |\b(static|public|private)\s+\w+[\w<>\[\]]*\s+\w+\s*\(|\w+\s+\w+\s*\([^;]*\)\s*\{",
                                   code))
    single_letter = len(re.findall(r"\b[a-zA-Z]\b\s*=", code))
    quality = 7 + (1 if comments else 0) + (1 if has_functions else -1) - min(2, long_lines) - (1 if single_letter > 8 else 0)
    edge = []
    if not re.search(r"if\s+not\b|len\(\w+\)\s*==\s*0|\.empty\(\)|isEmpty\(\)|==\s*0|<=\s*1|n\s*==\s*1", code):
        edge.append("Empty or single-element input is not explicitly handled.")
    if pass_rate < 1:
        edge.append("Some hidden test cases failed - re-check boundary conditions and duplicates.")
    return S.CodeReview(
        approach_quality=approach,
        approach_comment=f"Passed {passed}/{total} tests; estimated loop nesting depth {depth} (static analysis).",
        time_complexity=guess,
        space_complexity="O(n)" if re.search(r"\[\]|dict\(|set\(|\{\}|vector<|HashMap|ArrayList|list\(", code) else "O(1)",
        complexity_score=cscore,
        complexity_comment=f"Expected {expected_time}; estimated {guess}.",
        missed_edge_cases=edge,
        code_quality=_clamp(quality),
        code_quality_comment=f"{len(lines)} lines, {comments} comment line(s), "
                             f"{'uses' if has_functions else 'no'} helper functions.",
        better_approach="Review the hints for the intended technique." if cscore < 10 else "",
    )


# ----------------------------------------------------------------- resume --
SECTION_HEADINGS = {
    "education": ["education", "academic", "academics"],
    "experience": ["experience", "internship", "internships", "work history", "employment"],
    "projects": ["projects", "project", "academic projects", "personal projects"],
    "skills": ["skills", "technical skills", "tech stack", "technologies"],
    "certifications": ["certifications", "certification", "courses", "certificates"],
    "achievements": ["achievements", "awards", "honors", "honours", "accomplishments", "extracurricular"],
    "summary": ["summary", "objective", "profile", "about me"],
}


def split_sections(text: str) -> dict[str, list[str]]:
    """Split resume text into sections using common headings."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        key = re.sub(r"[^a-z ]", "", line.lower()).strip()
        found = None
        if len(key.split()) <= 3:
            for sec, heads in SECTION_HEADINGS.items():
                if key in heads:
                    found = sec
                    break
        if found:
            current = found
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return sections


def heuristic_resume_extraction(text: str, taxonomy_skills: list[str]) -> S.ResumeExtraction:
    sec = split_sections(text)
    header = sec.get("header", [])
    name = header[0] if header and len(header[0].split()) <= 5 and "@" not in header[0] else ""

    def bullets(lines: list[str]) -> list[str]:
        return [re.sub(r"^[\-•\*●▪>]+\s*", "", ln) for ln in lines]

    projects = []
    cur: S.ResumeProject | None = None
    for ln in sec.get("projects", []):
        is_bullet = bool(re.match(r"^[\-•\*●▪>]", ln))
        clean = bullets([ln])[0]
        if not is_bullet and len(clean.split()) <= 12:
            title = re.split(r"\s[|–\-:]\s|:", clean)[0].strip()
            cur = S.ResumeProject(name=title or clean, description="",
                                  tech=[s for s in taxonomy_skills if s.lower() in clean.lower()])
            projects.append(cur)
        elif cur is not None:
            cur.description = (cur.description + " " + clean).strip()
            cur.tech = sorted(set(cur.tech) | {s for s in taxonomy_skills if s.lower() in clean.lower()})
        else:
            cur = S.ResumeProject(name=clean[:60], description=clean)
            projects.append(cur)
    education = []
    edu_lines = sec.get("education", [])
    if edu_lines:
        education.append(S.ResumeEducation(
            institution=next((ln for ln in edu_lines if re.search(r"institute|university|college|school|iit|nit", ln, re.I)),
                             edu_lines[0]),
            degree=next((ln for ln in edu_lines if re.search(r"b\.?tech|b\.?e\b|bachelor|m\.?tech|master|b\.?sc|mca|bca", ln, re.I)), ""),
            year=(re.search(r"(19|20)\d{2}", " ".join(edu_lines)) or [""])[0],
            score=(re.search(r"(cgpa|gpa|percentage)[:\s]*[\d.]+", " ".join(edu_lines), re.I) or [""])[0],
        ))
    experience = []
    exp_lines = sec.get("experience", [])
    if exp_lines:
        experience.append(S.ResumeExperience(company=bullets(exp_lines[:1])[0][:80], role="",
                                             highlights=bullets(exp_lines[1:6])))
    return S.ResumeExtraction(
        name=name,
        skills=taxonomy_skills,
        projects=projects[:8],
        education=education,
        experience=experience,
        certifications=bullets(sec.get("certifications", []))[:10],
        achievements=bullets(sec.get("achievements", []))[:10],
    )


def heuristic_resume_suggestions(breakdown: dict) -> S.ResumeSuggestions:
    tips = {
        "keywords": "Add role-relevant keywords from the job description to your skills and project bullets.",
        "formatting": "Use clear section headings, consistent bullets and include email, phone and GitHub/LinkedIn links.",
        "impact": "Quantify outcomes in bullets (%, time saved, users, accuracy) using action verbs.",
        "structure": "Order sections as Education, Skills, Projects, Experience and keep it to one page (400-800 words).",
    }
    comps = sorted(breakdown.get("components", {}).items(), key=lambda kv: kv[1].get("score", 0) / max(1, kv[1].get("max", 1)))
    sug = [tips[k] for k, _ in comps if k in tips]
    sug += [f"Address: {d}" for k, v in comps for d in v.get("details", [])[:1] if v.get("score", 0) < v.get("max", 1) * 0.6]
    return S.ResumeSuggestions(summary=f"ATS score {breakdown.get('total', 0)}/100 computed by rule-based checks.",
                               suggestions=sug[:6] or ["Keep the resume updated with recent projects."])


def rewrite_bullet(bullet: str) -> S.BulletRewriteItem:
    b = re.sub(r"^[\-•\*]+\s*", "", bullet.strip())
    low = b.lower()
    new = b
    for weak, strong in STRONG_VERB_FOR.items():
        if low.startswith(weak) or low.startswith("i " + weak):
            new = strong + " " + b[low.index(weak) + len(weak):].strip()
            break
    else:
        if not any(low.startswith(v) for v in ACTION_VERBS):
            new = "Developed " + b[0].lower() + b[1:] if b else b
    new = new.rstrip(".")
    if not NUMBER_RE.search(new):
        new += ", improving [metric] by [X%]"
    return S.BulletRewriteItem(original=bullet, rewritten=new[0].upper() + new[1:] if new else new,
                               rationale="Leads with an action verb and adds a measurable-impact placeholder to fill in.")


def heuristic_cover_letter(name: str, company: str, role: str, skills: list[str], projects: list[str]) -> S.CoverLetter:
    sk = ", ".join(skills[:5]) or "software development"
    pj = projects[0] if projects else "my recent projects"
    body = (
        f"Dear Hiring Team at {company},\n\n"
        f"I am writing to apply for the {role} position. I am a computer science student who enjoys building "
        f"practical software, and my experience with {sk} matches the requirements you describe.\n\n"
        f"In {pj}, I worked end to end from design to deployment, which taught me to break problems down, "
        f"test carefully and communicate progress clearly with teammates.\n\n"
        f"I would welcome the chance to bring the same ownership to {company}. Thank you for your time and "
        f"consideration.\n\nSincerely,\n{name or 'Candidate'}"
    )
    return S.CoverLetter(subject=f"Application for {role} - {name or 'Candidate'}", body=body)


def heuristic_jd_reasoning(jd_skills: list[str], matched: list[str], missing: list[str], match_pct: float) -> S.JDMatchReasoning:
    return S.JDMatchReasoning(
        jd_skills=jd_skills,
        summary=f"Semantic skill match is {match_pct:.0f}%. {len(matched)} JD skill(s) are covered and "
                f"{len(missing)} are missing (rule-based summary).",
        strengths=[f"Demonstrated {s}" for s in matched[:5]],
        gaps=missing[:6],
        recommendations=[f"Build a small project or complete a course that uses {s}." for s in missing[:4]]
        or ["Tailor project bullets to mirror the JD's wording."],
    )


# --------------------------------------------------------------- feedback --
def heuristic_summary(kind: str, overall: float, results: list[dict]) -> S.PerformanceSummary:
    strengths, weaknesses = [], []
    for r in results:
        s = r.get("score") or 0
        label = r.get("title") or r.get("question", "")[:60]
        (strengths if s >= 70 else weaknesses).append(f"{label}: scored {s:.0f}/100.")
        for k, v in (r.get("rubric") or {}).items():
            if isinstance(v, (int, float)):
                if v >= 8:
                    strengths.append(f"Strong {k.replace('_', ' ')} ({v}/10) on '{label[:40]}'.")
                elif v <= 4:
                    weaknesses.append(f"Weak {k.replace('_', ' ')} ({v}/10) on '{label[:40]}'.")
    return S.PerformanceSummary(
        summary=f"{kind.upper()} session overall score {overall:.0f}/100 across {len(results)} question(s).",
        strengths=strengths[:5] or ["Completed the session and attempted every question."],
        weaknesses=weaknesses[:5] or ["No major weaknesses detected in this session."],
    )


def heuristic_skill_gaps(low_areas: list[str], taxonomy: dict[str, dict]) -> S.SkillGapMapping:
    gaps = []
    text = " ".join(low_areas).lower()
    for sid, info in taxonomy.items():
        keys = [info["name"].lower()] + [k.lower() for k in info.get("keywords", [])]
        if any(k and k in text for k in keys):
            gaps.append(S.SkillGap(skill_id=sid, reason="Matched a low-scoring area.", severity="medium"))
    return S.SkillGapMapping(skill_gaps=gaps[:6])


def heuristic_question_feedback(items: list[dict]) -> S.QuestionFeedback:
    out = []
    for it in items:
        if it.get("type") == "dsa":
            fb = (f"Passed {it.get('passed', 0)}/{it.get('total', 0)} tests. "
                  f"Estimated complexity {it.get('time_complexity', '?')} vs expected {it.get('expected_time', '?')}.")
            out.append(S.QuestionFeedbackItem(question=it.get("question", ""), feedback=fb))
        else:
            weak = [k for k, v in (it.get("rubric") or {}).items() if isinstance(v, (int, float)) and v <= 5]
            fb = ("Focus on improving " + ", ".join(w.replace("_", " ") for w in weak) + ". ") if weak else "Solid answer. "
            fb += "Use the STAR format and end with a measurable result."
            model = ("Situation: In my third-year team project our app missed a demo deadline. Task: As the "
                     "backend owner I had to recover the schedule. Action: I split the remaining work into daily "
                     "goals, paired with a teammate on the blocking API and set up automated tests. Result: We "
                     "shipped two days early, cut bug reports by 40%, and I learned to surface risks early.")
            out.append(S.QuestionFeedbackItem(question=it.get("question", ""), feedback=fb, model_answer=model))
    return S.QuestionFeedback(items=out)


def heuristic_improvement_plan(weaknesses: list[str], gaps: list[str]) -> S.ImprovementPlan:
    plan = [f"Spend 3 focused hours this week on {g}, then attempt one timed practice question." for g in gaps[:3]]
    plan += [f"Rework this weakness with a mock answer recorded aloud: {w}" for w in weaknesses[:2]]
    plan.append("Retake a PrepWise mock interview next week and compare the readiness score.")
    return S.ImprovementPlan(suggestions=plan)


# ---------------------------------------------------------------- roadmap --
def heuristic_roadmap(target_role: str, timeline_months: int, hours_per_week: int, skills: list[str],
                      skill_gaps: list[str], courses: list[dict], plan_weeks: int) -> S.RoadmapOutput:
    gaps = skill_gaps or ["dsa_fundamentals", "communication"]
    total_weeks = timeline_months * 4
    phase_len = max(1, total_weeks // 3)
    phases = [
        S.RoadmapPhase(name="Foundations", weeks=f"1-{phase_len}", goals=["Close the biggest skill gaps"], skills=gaps[:2]),
        S.RoadmapPhase(name="Build", weeks=f"{phase_len + 1}-{2 * phase_len}", goals=["Ship a portfolio project"],
                       skills=gaps[2:4] or gaps[:1]),
        S.RoadmapPhase(name="Interview mode", weeks=f"{2 * phase_len + 1}-{total_weeks}",
                       goals=["Weekly mock interviews", "Apply to roles"], skills=["interview_practice"]),
    ]
    weekly = []
    for w in range(1, plan_weeks + 1):
        g = gaps[(w - 1) % len(gaps)]
        weekly.append(S.RoadmapWeek(week=w, focus=g.replace("_", " ").title(),
                                    tasks=[f"Study {g.replace('_', ' ')} ({max(1, hours_per_week // 2)} h)",
                                           "Solve 5 practice problems", "Log learnings in your notes"]))
    recs = []
    for c in courses:
        recs.append(S.RoadmapCourse(course_id=c["id"], skill=(c.get("skills") or [""])[0], why="Matches a skill gap."))
        if len(recs) >= 5:
            break
    return S.RoadmapOutput(
        title=f"{timeline_months}-month roadmap to {target_role}",
        summary=f"Rule-based plan for {hours_per_week} h/week focusing on: {', '.join(gaps[:4])}.",
        phases=phases,
        weekly_plan=weekly,
        milestones=[S.RoadmapMilestone(week=phase_len, title="Foundations complete"),
                    S.RoadmapMilestone(week=2 * phase_len, title="Portfolio project deployed")],
        skills_to_learn=gaps,
        projects=[S.RoadmapProject(title=f"{target_role} portfolio project", description="End-to-end app using your target stack.",
                                   skills=gaps[:3])],
        recommended_courses=recs,
    )


def keyword_frequencies(text: str, top: int = 15) -> list[tuple[str, int]]:
    c = Counter(w.lower() for w in words(text) if w.lower() not in STOPWORDS and len(w) > 2)
    return c.most_common(top)
