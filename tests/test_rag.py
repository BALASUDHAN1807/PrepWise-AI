"""RAG retrieval, course validation and the hallucination guard over the curated question bank."""
import pytest

from backend.models import llm_schemas as S
from backend.services import question_service as Q
from backend.services import rag_service as R
from backend.services import roadmap_service as RM


def test_index_builds_both_collections():
    info = R.build_index(force=True)
    assert info["questions"] == len(R.load_question_bank()) >= 30
    assert info["courses"] == len(R.load_courses()) >= 25


def test_question_retrieval_respects_topic_filter():
    hits = R.retrieve_questions("shortest path in a weighted network", k=3, topic="graphs")
    assert hits and all(h["topic"] == "graphs" for h in hits)


def test_question_retrieval_ranks_semantically_close_first():
    hits = R.retrieve_questions("merge overlapping intervals of meetings", k=3)
    assert hits[0]["id"] == "arr-005"


def test_course_retrieval_only_returns_catalog_courses():
    catalog = R.course_ids()
    res = R.retrieve_courses(["dynamic_programming", "sql_databases", "communication", "nonexistent_skill"])
    assert res and all(c["id"] in catalog for c in res)
    assert any("dynamic_programming" in c["matched_skills"] for c in res)


def test_catalog_integrity():
    cs = R.load_courses()
    assert len(cs) >= 25 and {c["platform"] for c in cs} == {"Coursera", "Udemy", "NPTEL"}
    tax = R.load_taxonomy()
    for c in cs:
        assert set(c) >= {"id", "title", "platform", "skills", "level", "url"}
        assert c["url"].startswith("https://") and all(s in tax for s in c["skills"])
    assert len({c["id"] for c in cs}) == len(cs)


def test_roadmap_validation_rejects_invented_courses():
    catalog = {c["id"]: c for c in R.load_courses()}
    retrieved = {"coursera-dsa-ucsd", "nptel-dbms"}
    out = S.RoadmapOutput(
        title="t", summary="s", phases=[S.RoadmapPhase(name="p", weeks="1-4")],
        weekly_plan=[S.RoadmapWeek(week=1, focus="f", tasks=["t"])],
        recommended_courses=[S.RoadmapCourse(course_id="coursera-dsa-ucsd"),
                             S.RoadmapCourse(course_id="udemy-totally-invented"),   # not in catalog
                             S.RoadmapCourse(course_id="udemy-sql-bootcamp"),       # real but not retrieved
                             S.RoadmapCourse(course_id="coursera-dsa-ucsd")])       # duplicate
    valid, rejected = RM.validate_roadmap_courses(out, retrieved, catalog)
    assert [v["course_id"] for v in valid] == ["coursera-dsa-ucsd"]
    assert set(rejected) == {"udemy-totally-invented", "udemy-sql-bootcamp"}
    assert valid[0]["url"] == catalog["coursera-dsa-ucsd"]["url"]  # url comes from the catalog, never the LLM


def test_build_roadmap_uses_only_catalog_courses():
    gen = RM.build_roadmap("Backend Engineer", 3, 8, ["Python"], ["graphs", "sql_databases"])
    catalog = R.course_ids()
    assert gen["content"]["recommended_courses"]
    assert all(c["course_id"] in catalog for c in gen["content"]["recommended_courses"])


@pytest.mark.parametrize("qid", [q["id"] for q in R.load_question_bank()])
def test_every_bank_question_passes_the_hallucination_guard(qid):
    q = next(x for x in R.load_question_bank() if x["id"] == qid)
    g = Q.guard_question(Q._strip(q))
    assert g["ok"], f"{qid}: {g}"
    assert len(q["test_cases"]) >= 6 and len(q["hints"]) == 3


def test_bank_has_required_topic_coverage():
    topics = {q["topic"] for q in R.load_question_bank()}
    assert topics >= {"arrays", "strings", "linked_lists", "trees", "graphs", "dp", "sql"}
    assert len(R.load_question_bank()) >= 30


def test_guard_rejects_a_wrong_reference_solution():
    q = Q._strip(next(x for x in R.load_question_bank() if x["id"] == "arr-006"))
    q["reference_solution"] = "print(0)"
    g = Q.guard_question(q)
    assert not g["ok"] and "reference solution failed" in g["reason"]


def test_guard_rejects_too_few_or_duplicate_tests():
    q = Q._strip(next(x for x in R.load_question_bank() if x["id"] == "arr-006"))
    assert not Q.guard_question({**q, "test_cases": q["test_cases"][:4]})["ok"]
    dup = [dict(q["test_cases"][0]) for _ in range(6)]
    assert "duplicate" in Q.guard_question({**q, "test_cases": dup})["reason"]


def test_generate_regenerates_when_guard_fails(monkeypatch):
    """First LLM question has a broken reference solution -> rejected; second is accepted."""
    from backend.llm.client import LLMJSONResult

    good = Q._strip(next(x for x in R.load_question_bank() if x["id"] == "str-001"))
    bad = {**good, "reference_solution": "print('NO')\nprint('x')"}
    queue = [bad, good]

    def fake_run_prompt(name, variables, **kw):
        return LLMJSONResult(data=S.GeneratedQuestion(**queue.pop(0)), provider="fake", model="m", latency_ms=1, attempts=1)

    monkeypatch.setattr(Q, "run_prompt", fake_run_prompt)
    out = Q.generate_question("strings", "easy")
    assert out["meta"]["attempts"] == 2 and out["meta"]["source"] == "llm"
    assert out["meta"]["rejections"] and out["meta"]["guard"]["ok"]


def test_generate_falls_back_to_bank_after_three_bad_attempts(monkeypatch):
    from backend.llm.client import LLMJSONResult

    good = Q._strip(next(x for x in R.load_question_bank() if x["id"] == "str-001"))
    bad = {**good, "reference_solution": "print('NO')\nprint('x')"}
    monkeypatch.setattr(Q, "run_prompt", lambda *a, **k: LLMJSONResult(
        data=S.GeneratedQuestion(**bad), provider="fake", model="m", latency_ms=1, attempts=1))
    out = Q.generate_question("strings", "easy")
    assert out["meta"]["source"] == "bank" and out["meta"]["attempts"] == 3 and out["meta"]["guard"]["ok"]


def test_public_view_hides_hidden_tests_and_solution():
    q = Q._strip(next(x for x in R.load_question_bank() if x["id"] == "dp-002"))
    pub = Q.public_view(q)
    assert "reference_solution" not in pub and "test_cases" not in pub and "hints" not in pub
    assert len(pub["visible_tests"]) == 2 and pub["hidden_test_count"] == len(q["test_cases"]) - 2
