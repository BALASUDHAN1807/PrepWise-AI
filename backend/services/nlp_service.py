"""NLP utilities: spaCy pipeline, keyword/entity extraction, sentiment and speech analytics.

spaCy usage (Unit 2): tokenization, sentence splitting, named-entity recognition
(``en_core_web_sm``) and rule-based skill matching with ``PhraseMatcher`` against
``data/skills_taxonomy.json``.

If the statistical model is not installed, the service falls back to
``spacy.blank("en")`` + ``sentencizer`` (tokenization and phrase matching still work,
NER does not) and reports which mode is active via :func:`nlp_status`.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from backend.core.config import get_settings

log = logging.getLogger("prepwise.nlp")

SKILL_ALIASES = {
    "js": "JavaScript", "javascript": "JavaScript", "react": "React", "reactjs": "React", "react.js": "React", "node": "Node.js",
    "nodejs": "Node.js", "node.js": "Node.js", "py": "Python", "c++": "C++", "cpp": "C++", "mongo": "MongoDB",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "mysql": "MySQL", "sql": "SQL", "java": "Java",
    "python": "Python", "html": "HTML", "css": "CSS", "git": "Git", "github": "GitHub", "docker": "Docker",
    "kubernetes": "Kubernetes", "aws": "AWS", "azure": "Azure", "gcp": "GCP", "linux": "Linux", "fastapi": "FastAPI",
    "flask": "Flask", "django": "Django", "streamlit": "Streamlit", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "scikit-learn": "scikit-learn", "sklearn": "scikit-learn", "pandas": "Pandas", "numpy": "NumPy",
    "machine learning": "Machine Learning", "deep learning": "Deep Learning", "nlp": "NLP", "llm": "LLMs",
    "llms": "LLMs", "typescript": "TypeScript", "redis": "Redis", "kafka": "Kafka",
    "rest api": "REST APIs", "rest apis": "REST APIs", "restful": "REST APIs", "express.js": "Express", "expressjs": "Express", "apache spark": "Spark", "pyspark": "Spark", "ms excel": "Excel", "microsoft excel": "Excel", "graphql": "GraphQL", "c": "C", "dsa": "Data Structures",
    "data structures": "Data Structures", "algorithms": "Algorithms", "oop": "OOP", "system design": "System Design",
    "ci/cd": "CI/CD", "tailwind": "Tailwind CSS", "next.js": "Next.js", "nextjs": "Next.js", "vue": "Vue",
    "angular": "Angular", "spring": "Spring", "spring boot": "Spring Boot", "hadoop": "Hadoop", 
    "tableau": "Tableau", "power bi": "Power BI", "figma": "Figma", "selenium": "Selenium",
    "junit": "JUnit", "jira": "Jira", "cybersecurity": "Cybersecurity", 
    "computer networks": "Computer Networks", "rag": "RAG", "prompt engineering": "Prompt Engineering",
}


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy once. Returns (nlp, mode) where mode is 'model' or 'blank'."""
    import spacy

    name = get_settings().spacy_model
    try:
        nlp = spacy.load(name)
        mode = "model"
    except Exception:  # model package missing / incompatible
        log.warning("spaCy model '%s' not available; using blank English pipeline (no NER)", name)
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        mode = "blank"
    from spacy.matcher import PhraseMatcher

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    terms = sorted(SKILL_ALIASES)
    matcher.add("SKILL", [nlp.make_doc(t) for t in terms])
    nlp.meta["prepwise_matcher"] = matcher
    return nlp, mode


def nlp_status() -> dict:
    nlp, mode = get_nlp()
    return {"spacy_mode": mode, "pipeline": nlp.pipe_names, "ner": "ner" in nlp.pipe_names}


def extract_skills(text: str) -> list[str]:
    """Skills mentioned in ``text`` using PhraseMatcher (spaCy) + alias normalisation."""
    nlp, _ = get_nlp()
    doc = nlp.make_doc(text or "")
    matcher = nlp.meta["prepwise_matcher"]
    found: dict[str, None] = {}
    for _id, start, end in matcher(doc):
        key = doc[start:end].text.lower()
        if key == "c" and doc[start:end].text != "C":  # avoid matching the letter "c" in prose
            continue
        found[SKILL_ALIASES.get(key, doc[start:end].text)] = None
    return list(found)


def extract_entities(text: str) -> dict[str, list[str]]:
    """Named entities useful for resume parsing (organisations, dates, locations)."""
    nlp, mode = get_nlp()
    out: dict[str, list[str]] = {"orgs": [], "dates": [], "locations": [], "persons": []}
    if mode == "model":
        doc = nlp((text or "")[:100_000])
        label_map = {"ORG": "orgs", "DATE": "dates", "GPE": "locations", "PERSON": "persons"}
        for ent in doc.ents:
            key = label_map.get(ent.label_)
            if key and ent.text not in out[key]:
                out[key].append(ent.text)
    # Regex fallback / supplement works in both modes.
    for m in re.finditer(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b|\b(?:19|20)\d{2}\s*[-–]\s*(?:(?:19|20)\d{2}|Present|present)\b", text or ""):
        if m.group(0) not in out["dates"]:
            out["dates"].append(m.group(0))
    for m in re.finditer(r"\b[A-Z][A-Za-z&. ]{2,40}(?:Institute|University|College|Technologies|Technology|Solutions|Labs|Pvt\.? Ltd\.?|Inc\.?)\b", text or ""):
        if m.group(0).strip() not in out["orgs"]:
            out["orgs"].append(m.group(0).strip())
    for k in out:
        out[k] = out[k][:15]
    return out


def keywords(text: str, top: int = 15) -> list[str]:
    """Top noun-like keywords (lower-cased lemmas are approximated by lower-casing)."""
    nlp, _ = get_nlp()
    doc = nlp.make_doc(text or "")
    from collections import Counter

    from backend.services.heuristics import STOPWORDS

    toks = [t.text.lower() for t in doc if t.is_alpha and len(t.text) > 2 and t.text.lower() not in STOPWORDS]
    return [w for w, _ in Counter(toks).most_common(top)]


def sentence_count(text: str) -> int:
    nlp, _ = get_nlp()
    return max(1, len(list(nlp(text or "").sents))) if text else 0


# ---------------------------------------------------------------- analytics --
@lru_cache(maxsize=1)
def load_fillers() -> list[str]:
    path = get_settings().data_path / "filler_words.txt"
    try:
        lines = [ln.strip().lower() for ln in path.read_text(encoding="utf-8").splitlines()]
    except OSError:
        from backend.services.heuristics import FILLERS_DEFAULT

        return FILLERS_DEFAULT
    return [ln for ln in lines if ln and not ln.startswith("#")]


POSITIVE = {"improved", "achieved", "learned", "success", "successfully", "delivered", "won", "proud", "enjoy",
            "excited", "collaborated", "solved", "opportunity", "grew", "confident", "great", "helped", "resolved"}
NEGATIVE = {"failed", "fail", "problem", "struggled", "difficult", "mistake", "worst", "hate", "angry", "late",
            "stress", "stressed", "nervous", "confused", "blame", "never", "unable", "bad"}


@lru_cache(maxsize=1)
def _transformer_sentiment():
    """Load the Hugging Face sentiment pipeline once; None if unavailable (offline etc.)."""
    s = get_settings()
    if not s.enable_transformer_sentiment:
        return None
    try:
        from transformers import pipeline

        return pipeline("sentiment-analysis", model=s.sentiment_model)
    except Exception as exc:  # no weights / offline / transformers missing
        log.warning("Transformer sentiment unavailable (%s); using lexicon sentiment", type(exc).__name__)
        return None


def sentiment(text: str) -> dict:
    """Tone of an answer. Uses a transformer model when available, else a small lexicon."""
    pipe = _transformer_sentiment()
    if pipe is not None and text.strip():
        try:
            r = pipe(text[:1500])[0]
            score = r["score"] if r["label"].upper().startswith("POS") else -r["score"]
            label = "positive" if score > 0.25 else "negative" if score < -0.25 else "neutral"
            return {"label": label, "score": round(float(score), 3), "method": "transformer"}
        except Exception as exc:  # pragma: no cover - runtime model failure
            log.warning("sentiment pipeline failed: %s", exc)
    toks = [t.lower() for t in re.findall(r"[A-Za-z']+", text)]
    pos, neg = sum(t in POSITIVE for t in toks), sum(t in NEGATIVE for t in toks)
    score = (pos - neg) / max(1, pos + neg) if (pos + neg) else 0.0
    label = "positive" if score > 0.25 else "negative" if score < -0.25 else "neutral"
    return {"label": label, "score": round(score, 3), "method": "lexicon"}


def speech_analytics(text: str, duration_sec: float | None = None, segments: list[dict] | None = None) -> dict:
    """Words per minute, filler words, pause estimate and sentiment/tone.

    ``duration_sec`` and ``segments`` (start/end seconds per Whisper segment) are only
    available for voice answers. For typed answers WPM and pauses are ``None``.
    """
    from backend.services.heuristics import count_fillers, words

    wc = len(words(text))
    fillers = count_fillers(text, load_fillers())
    wpm = round(wc / (duration_sec / 60), 1) if duration_sec and duration_sec > 0 else None
    pauses = None
    if segments and len(segments) > 1:
        gaps = [segments[i + 1]["start"] - segments[i]["end"] for i in range(len(segments) - 1)]
        long_gaps = [g for g in gaps if g >= 0.7]
        pauses = {"count": len(long_gaps), "total_sec": round(sum(long_gaps), 1),
                  "longest_sec": round(max(long_gaps), 1) if long_gaps else 0.0}
    pace = None
    if wpm is not None:
        pace = "too slow" if wpm < 100 else "too fast" if wpm > 180 else "good pace"
    return {
        "word_count": wc,
        "sentence_count": sentence_count(text),
        "wpm": wpm,
        "pace": pace,
        "filler_count": sum(fillers.values()),
        "fillers": fillers,
        "filler_rate_per_100_words": round(100 * sum(fillers.values()) / wc, 1) if wc else 0.0,
        "pauses": pauses,
        "sentiment": sentiment(text),
        "source": "voice" if duration_sec else "text",
    }


def dump_status() -> str:  # pragma: no cover - debugging helper
    return json.dumps(nlp_status())
