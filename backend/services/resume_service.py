"""Resume processing: text extraction, structured parsing, ATS score, JD match, rewriting.

Privacy: raw resume text is never written to logs; it is stored only in the user's own
``resumes`` row (deleted with the account) and sent to the configured LLM provider.
"""
from __future__ import annotations

import io
import re

from backend.llm.client import LLMClient
from backend.llm.runner import run_prompt
from backend.models import llm_schemas as S
from backend.services import heuristics as H
from backend.services import nlp_service as N
from backend.services import rag_service as R

MAX_RESUME_CHARS = 20_000
BULLET_RE = re.compile(r"^\s*[\-•\*●▪>]\s+")

ROLE_KEYWORDS = {
    "default": ["Python", "Java", "JavaScript", "SQL", "Git", "Data Structures", "Algorithms", "REST APIs", "OOP", "Linux"],
    "backend": ["Python", "Java", "Node.js", "SQL", "REST APIs", "Docker", "PostgreSQL", "MongoDB", "Git", "System Design"],
    "frontend": ["JavaScript", "React", "HTML", "CSS", "TypeScript", "Git", "REST APIs", "Tailwind CSS", "Figma", "Next.js"],
    "data": ["Python", "SQL", "Pandas", "NumPy", "Machine Learning", "scikit-learn", "Tableau", "Power BI", "Excel", "Statistics"],
    "ml": ["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "NLP", "scikit-learn", "Pandas", "LLMs", "RAG"],
    "devops": ["Docker", "Kubernetes", "Linux", "CI/CD", "AWS", "Git", "Python", "Terraform", "Networking", "Bash"],
    "full": ["JavaScript", "React", "Node.js", "SQL", "MongoDB", "REST APIs", "Git", "Docker", "HTML", "CSS"],
}


class ResumeError(ValueError):
    """User-facing problem with an uploaded resume."""


# ------------------------------------------------------------ extraction ----
def extract_text(filename: str, data: bytes) -> str:
    """Return plain text from a PDF or DOCX upload, with friendly errors."""
    if not data:
        raise ResumeError("The uploaded file is empty.")
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            import pdfplumber

            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        elif name.endswith(".docx"):
            import docx

            d = docx.Document(io.BytesIO(data))
            parts = [p.text for p in d.paragraphs]
            for table in d.tables:
                for row in table.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            text = "\n".join(parts)
        elif name.endswith(".txt"):
            text = data.decode("utf-8", "replace")
        else:
            raise ResumeError("Unsupported file type. Please upload a PDF or DOCX file.")
    except ResumeError:
        raise
    except Exception as exc:  # corrupt/encrypted files
        raise ResumeError(f"Could not read this file ({type(exc).__name__}). Is it corrupted or password protected?") from exc
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 80:
        raise ResumeError("No readable text was found. Scanned/image-only PDFs are not supported (no OCR); "
                          "please upload a text-based PDF or a DOCX.")
    return text[:MAX_RESUME_CHARS]


# --------------------------------------------------------------- parsing ----
def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for s in items:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def parse_resume(text: str, client: LLMClient | None = None) -> tuple[S.ResumeExtraction, dict]:
    """spaCy pre-extraction (entities + skills) fed into the LLM extraction prompt."""
    ents = N.extract_entities(text)
    skills = N.extract_skills(text)
    ctx = {"text": text, "taxonomy_skills": skills}
    res = run_prompt(
        "resume_extraction",
        {"resume_text": text, "orgs": ents["orgs"], "dates": ents["dates"], "taxonomy_skills": skills, "_ctx": ctx},
        fallback=lambda: H.heuristic_resume_extraction(text, skills),
        client=client,
    )
    parsed = res.data
    parsed.skills = _dedupe(list(parsed.skills) + skills)  # union of LLM and spaCy skills
    return parsed, {**res.info(), "spacy": N.nlp_status()}


def resume_bullets(text: str) -> list[str]:
    return [BULLET_RE.sub("", ln).strip() for ln in text.splitlines() if BULLET_RE.match(ln) and len(ln.split()) >= 4]


# ------------------------------------------------------------------- ATS ----
def _role_family(target_role: str) -> str:
    t = target_role.lower()
    for key, hints in [("ml", ["machine learning", "ml ", "ai ", "nlp", "deep learning"]), ("data", ["data", "analyst"]),
                       ("devops", ["devops", "sre", "cloud"]), ("frontend", ["front"]), ("backend", ["back"]),
                       ("full", ["full"])]:
        if any(h in t + " " for h in hints):
            return key
    return "default"


def ats_score(text: str, parsed: S.ResumeExtraction, target_role: str = "Software Engineer") -> dict:
    """Deterministic ATS score out of 100 (keywords 30, formatting 25, impact 25, structure 20).

    The LLM never produces this number; it only explains it (see ``resume_suggestions``).
    """
    comps: dict[str, dict] = {}
    low = text.lower()
    # --- keywords (30)
    wanted = ROLE_KEYWORDS[_role_family(target_role)]
    have = {s.lower() for s in parsed.skills} | {s.lower() for s in N.extract_skills(text)}
    hit = [k for k in wanted if k.lower() in have or k.lower() in low]
    kw = 30 * len(hit) / len(wanted)
    comps["keywords"] = {"score": round(kw, 1), "max": 30,
                         "details": [f"{len(hit)}/{len(wanted)} role keywords found ({', '.join(hit) or 'none'})"]
                         + ([f"Missing: {', '.join(k for k in wanted if k not in hit)}"] if len(hit) < len(wanted) else [])}
    # --- formatting (25)
    wc = len(text.split())
    fmt, fd = 0.0, []
    for ok, pts, good, bad in [
        (bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)), 5, "Email present", "Add a professional email address"),
        (bool(re.search(r"\+?\d[\d\s-]{8,}", text)), 4, "Phone number present", "Add a phone number"),
        (bool(re.search(r"linkedin\.com|github\.com", low)), 4, "LinkedIn/GitHub link present", "Add LinkedIn/GitHub links"),
        (bool(resume_bullets(text)), 4, "Uses bullet points", "Use bullet points for projects/experience"),
        (300 <= wc <= 900, 5, f"Length {wc} words is appropriate", f"Length {wc} words: aim for 300-900 (one page)"),
        (len(H.split_sections(text)) >= 4, 3, "Clear section headings detected", "Use standard headings (Education, Skills, Projects)"),
    ]:
        fmt += pts if ok else 0
        fd.append(good if ok else bad)
    comps["formatting"] = {"score": fmt, "max": 25, "details": fd}
    # --- impact (25)
    bullets = resume_bullets(text) or [ln for ln in text.splitlines() if len(ln.split()) >= 8]
    if bullets:
        metric = sum(bool(H.NUMBER_RE.search(b)) and bool(re.search(r"\d", b)) for b in bullets) / len(bullets)
        verb = sum(any(b.lower().startswith(v) for v in H.ACTION_VERBS) for b in bullets) / len(bullets)
        weak = sum(b.lower().startswith(("worked on", "responsible for", "helped", "involved in")) for b in bullets) / len(bullets)
    else:
        metric = verb = weak = 0.0
    imp = 25 * min(1.0, 0.6 * min(1, metric / 0.5) + 0.4 * min(1, verb / 0.6)) - 5 * weak
    comps["impact"] = {"score": round(max(0, imp), 1), "max": 25,
                       "details": [f"{metric:.0%} of bullets contain numbers/metrics (aim for 50%+)",
                                   f"{verb:.0%} of bullets start with a strong action verb (aim for 60%+)"]
                       + ([f"{weak:.0%} of bullets start with weak phrases like 'responsible for'"] if weak else [])}
    # --- structure (20)
    secs = H.split_sections(text)
    needed = {"education": 5, "skills": 5, "projects": 5, "experience": 3, "certifications": 1, "achievements": 1}
    st, sd = 0, []
    for sec, pts in needed.items():
        present = bool(secs.get(sec))
        st += pts if present else 0
        if not present and pts >= 3:
            sd.append(f"Missing section: {sec}")
    comps["structure"] = {"score": st, "max": 20, "details": sd or ["All key sections present"]}
    total = round(sum(c["score"] for c in comps.values()), 1)
    return {"total": min(100.0, total), "components": comps, "word_count": wc, "method": "rule-based (deterministic)"}


def ats_suggestions(breakdown: dict, parsed: S.ResumeExtraction, target_role: str, client: LLMClient | None = None) -> dict:
    res = run_prompt(
        "resume_suggestions",
        {"target_role": target_role, "ats_breakdown": breakdown["components"],
         "parsed_resume": parsed.model_dump_json()[:3000], "_ctx": {"breakdown": breakdown}},
        fallback=lambda: H.heuristic_resume_suggestions(breakdown), client=client)
    return {**res.data.model_dump(), "llm": res.info()}


# --------------------------------------------------------------- JD match ----
def jd_skills_from_text(jd: str) -> list[str]:
    skills = N.extract_skills(jd)
    tax = R.load_taxonomy()
    low = jd.lower()
    for info in tax.values():
        if info["category"] == "technical" and info["name"].lower() in low and info["name"] not in skills:
            skills.append(info["name"])
    return skills


def jd_match(parsed: S.ResumeExtraction, resume_text: str, jd: str, client: LLMClient | None = None) -> dict:
    """Match % = JD skills covered by the resume, using exact aliases then embedding similarity."""
    jd_sk = jd_skills_from_text(jd)
    if not jd_sk:
        raise ResumeError("No recognisable skills were found in the job description. Paste the full JD text.")
    resume_terms = _dedupe(parsed.skills + N.extract_skills(resume_text)
                           + [t for p in parsed.projects for t in p.tech])
    ctx_snips = resume_terms + [p.description[:200] for p in parsed.projects if p.description]
    info = R.embedding_info()
    thresh = 0.58 if info["semantic"] else 0.85
    jd_emb, res_emb = R.embed(jd_sk), R.embed(ctx_snips or ["none"])
    sims = R.cosine_matrix(jd_emb, res_emb)
    lowered = {t.lower() for t in resume_terms}
    matched, missing, detail = [], [], []
    for i, s in enumerate(jd_sk):
        best = float(sims[i].max()) if sims.size else 0.0
        exact = s.lower() in lowered
        ok = exact or best >= thresh
        (matched if ok else missing).append(s)
        detail.append({"skill": s, "matched": ok, "exact": exact, "similarity": round(best, 3)})
    pct = round(100 * len(matched) / len(jd_sk), 1)
    doc_sim = float(R.cosine_matrix(R.embed([jd]), R.embed([resume_text[:3000]]))[0, 0])
    res = run_prompt(
        "jd_match",
        {"jd_text": jd[:4000], "resume_skills": resume_terms, "resume_projects": [p.name for p in parsed.projects],
         "matched": matched, "missing": missing, "match_pct": pct,
         "_ctx": {"jd_skills": jd_sk, "matched": matched, "missing": missing, "match_pct": pct}},
        fallback=lambda: H.heuristic_jd_reasoning(jd_sk, matched, missing, pct), client=client)
    return {"match_pct": pct, "matched_skills": matched, "missing_skills": missing, "details": detail,
            "document_similarity": round(doc_sim, 3), "embedding": info, "reasoning": res.data.model_dump(),
            "llm": res.info()}


# --------------------------------------------------- rewriter / cover letter ----
def rewrite_bullets(bullets: list[str], target_role: str, client: LLMClient | None = None) -> dict:
    bullets = [b.strip() for b in bullets if b.strip()][:10]
    if not bullets:
        raise ResumeError("Provide at least one bullet point to rewrite.")
    res = run_prompt(
        "bullet_rewrite", {"target_role": target_role, "bullets": bullets, "_ctx": {"bullets": bullets}},
        fallback=lambda: S.BulletRewrite(items=[H.rewrite_bullet(b) for b in bullets]), client=client)
    return {**res.data.model_dump(), "llm": res.info()}


def cover_letter(parsed: S.ResumeExtraction, jd: str, company: str, role: str, tone: str = "professional",
                 client: LLMClient | None = None) -> dict:
    ctx = {"name": parsed.name, "company": company, "role": role, "skills": parsed.skills,
           "projects": [p.name for p in parsed.projects]}
    res = run_prompt(
        "cover_letter",
        {"name": parsed.name or "the candidate", "company": company, "role": role, "tone": tone,
         "parsed_resume": parsed.model_dump_json()[:3500], "jd_text": jd[:3500], "_ctx": ctx},
        fallback=lambda: H.heuristic_cover_letter(parsed.name, company, role, parsed.skills,
                                                  [p.name for p in parsed.projects]), client=client)
    return {**res.data.model_dump(), "llm": res.info()}
