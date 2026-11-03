"""Retrieval-Augmented Generation over local curated data.

Two collections are indexed:

* ``questions`` - ``data/question_bank.json`` (grounds DSA question generation)
* ``courses``   - ``data/courses.json`` (the ONLY source of course recommendations)

Embeddings (Unit 2): ``sentence-transformers`` (``all-MiniLM-L6-v2``) by default. If the
model cannot be loaded (offline sandbox, no weights) the service transparently falls back
to a deterministic hashing-vector embedding (bag of word and bigram hashes, L2-normalised)
so retrieval still works; :func:`embedding_info` reports which one is active.

Storage: ChromaDB (persistent) by default, or an in-memory numpy store (``VECTOR_STORE=memory``)
which tests use.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from backend.core.config import get_settings

log = logging.getLogger("prepwise.rag")
HASH_DIM = 512


# ------------------------------------------------------------------- data ---
@lru_cache(maxsize=1)
def load_question_bank() -> list[dict]:
    """Questions from data/question_bank.json (extend this file with your own)."""
    return json.loads((get_settings().data_path / "question_bank.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_courses() -> list[dict]:
    return json.loads((get_settings().data_path / "courses.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, dict]:
    return json.loads((get_settings().data_path / "skills_taxonomy.json").read_text(encoding="utf-8"))


def course_ids() -> set[str]:
    return {c["id"] for c in load_courses()}


def clear_caches() -> None:
    for fn in (load_question_bank, load_courses, load_taxonomy, _embedder, get_store):
        fn.cache_clear()


# ------------------------------------------------------------- embeddings ---
class HashEmbedder:
    """Deterministic fallback: hashed unigram+bigram counts, L2 normalised."""

    name = "hash-bow-512"

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for t in texts:
            v = np.zeros(HASH_DIM, dtype=np.float32)
            toks = re.findall(r"[a-z0-9+#.]+", t.lower())
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for g in grams:
                h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
                v[h % HASH_DIM] += 1.0 if "_" not in g else 0.7
            n = np.linalg.norm(v)
            rows.append(v / n if n else v)
        return np.vstack(rows) if rows else np.zeros((0, HASH_DIM), dtype=np.float32)


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


@lru_cache(maxsize=1)
def _embedder():
    s = get_settings()
    if s.embedding_backend == "sentence-transformers":
        try:
            return STEmbedder(s.embedding_model)
        except Exception as exc:
            log.warning("sentence-transformers unavailable (%s); falling back to hashing embeddings",
                        type(exc).__name__)
    return HashEmbedder()


def embed(texts: Iterable[str]) -> np.ndarray:
    return _embedder().encode(list(texts))


def embedding_info() -> dict:
    e = _embedder()
    return {"embedder": e.name, "semantic": not isinstance(e, HashEmbedder)}


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of ``a`` and rows of ``b`` (inputs assumed normalised)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return a @ b.T


# ----------------------------------------------------------------- stores ---
class MemoryStore:
    """Tiny numpy vector store with the same interface as the Chroma wrapper."""

    def __init__(self):
        self.cols: dict[str, dict] = {}

    def upsert(self, collection: str, ids: list[str], docs: list[str], metas: list[dict]) -> None:
        self.cols[collection] = {"ids": ids, "docs": docs, "metas": metas, "emb": embed(docs)}

    def count(self, collection: str) -> int:
        return len(self.cols.get(collection, {}).get("ids", []))

    def query(self, collection: str, text: str, k: int, where: dict | None = None) -> list[dict]:
        col = self.cols.get(collection)
        if not col:
            return []
        sims = (col["emb"] @ embed([text])[0]).tolist()
        order = sorted(range(len(sims)), key=lambda i: -sims[i])
        out = []
        for i in order:
            m = col["metas"][i]
            if where and any(m.get(k_) != v for k_, v in where.items()):
                continue
            out.append({"id": col["ids"][i], "score": float(sims[i]), "meta": m})
            if len(out) == k:
                break
        return out


class ChromaStore:
    def __init__(self, path: str):
        import chromadb

        self.client = chromadb.PersistentClient(path=path)

    def upsert(self, collection: str, ids, docs, metas) -> None:
        try:
            self.client.delete_collection(collection)
        except Exception:
            pass
        col = self.client.create_collection(collection, metadata={"hnsw:space": "cosine"})
        col.add(ids=ids, documents=docs, embeddings=embed(docs).tolist(), metadatas=metas)

    def count(self, collection: str) -> int:
        try:
            return self.client.get_collection(collection).count()
        except Exception:
            return 0

    def query(self, collection: str, text: str, k: int, where: dict | None = None) -> list[dict]:
        try:
            col = self.client.get_collection(collection)
        except Exception:
            return []
        res = col.query(query_embeddings=embed([text]).tolist(), n_results=min(k, max(1, col.count())),
                        where=where or None)
        return [{"id": i, "score": 1 - d, "meta": m}
                for i, d, m in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])]


@lru_cache(maxsize=1)
def get_store():
    s = get_settings()
    if s.vector_store == "chroma":
        try:
            return ChromaStore(s.chroma_dir)
        except Exception as exc:
            log.warning("Chroma unavailable (%s); using in-memory store", exc)
    return MemoryStore()


# ---------------------------------------------------------------- indexing ---
def _question_doc(q: dict) -> str:
    return f"{q['title']}. {q['topic']} {q['difficulty']}. {q['statement']}"


def _course_doc(c: dict) -> str:
    tax = load_taxonomy()
    names = " ".join(tax[s]["name"] for s in c["skills"] if s in tax)
    return f"{c['title']}. Skills: {names}. Level: {c['level']}. Platform: {c['platform']}."


def build_index(force: bool = False) -> dict:
    """(Re)build both collections from the JSON files. Idempotent."""
    store = get_store()
    qs, cs = load_question_bank(), load_courses()
    if force or store.count("questions") != len(qs):
        store.upsert("questions", [q["id"] for q in qs], [_question_doc(q) for q in qs],
                     [{"topic": q["topic"], "difficulty": q["difficulty"], "kind": q.get("kind", "code")} for q in qs])
    if force or store.count("courses") != len(cs):
        store.upsert("courses", [c["id"] for c in cs], [_course_doc(c) for c in cs],
                     [{"platform": c["platform"], "level": c["level"]} for c in cs])
    return {"questions": store.count("questions"), "courses": store.count("courses"), **embedding_info()}


# --------------------------------------------------------------- retrieval ---
def retrieve_questions(query: str, k: int = 3, topic: str | None = None, kind: str | None = None) -> list[dict]:
    """Similar bank questions (used to ground the style of newly generated questions)."""
    build_index()
    where = {}
    if topic:
        where["topic"] = topic
    if kind:
        where["kind"] = kind
    by_id = {q["id"]: q for q in load_question_bank()}
    hits = get_store().query("questions", query, k, where or None)
    return [{**by_id[h["id"]], "_score": round(h["score"], 3)} for h in hits if h["id"] in by_id]


def retrieve_courses(skill_ids: list[str], k_per_skill: int = 3, level: str | None = None) -> list[dict]:
    """Courses for skill gaps: tag match first (exact), semantic similarity to break ties.

    Only entries from courses.json are ever returned, which is what lets the roadmap
    validator reject any course id the LLM invents.
    """
    build_index()
    tax, courses = load_taxonomy(), load_courses()
    chosen: dict[str, dict] = {}
    for sid in skill_ids:
        name = tax.get(sid, {}).get("name", sid.replace("_", " "))
        tagged = [c for c in courses if sid in c["skills"]]
        if level:
            tagged = [c for c in tagged if c["level"] == level] or tagged
        sem = {h["id"]: h["score"] for h in get_store().query("courses", name, len(courses))}
        tagged.sort(key=lambda c: -sem.get(c["id"], 0))
        picked = tagged[:k_per_skill]
        if not picked:  # no tag match -> best semantic neighbours
            picked = [next(c for c in courses if c["id"] == h["id"])
                      for h in get_store().query("courses", name, 2)]
        for c in picked:
            chosen.setdefault(c["id"], {**c, "matched_skills": []})["matched_skills"].append(sid)
    return list(chosen.values())
