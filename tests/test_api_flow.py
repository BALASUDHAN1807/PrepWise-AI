"""End-to-end API tests (mock LLM provider, hash embeddings, in-memory vector store)."""
import io
from datetime import timedelta

import docx
import pytest
from reportlab.pdfgen import canvas
from sqlalchemy import select

from backend.core.database import SessionLocal
from backend.models import db_models as M
from backend.services import rag_service as R
from tests.conftest import signup

RESUME_LINES = [
    "Asha Krishnan", "asha.k@example.com | +91 98765 43210 | github.com/asha | linkedin.com/in/asha",
    "Education", "B.Tech Computer Science and Engineering, SRM Institute of Science and Technology, 2028, CGPA 8.9",
    "Skills", "Python, Java, SQL, React, Node.js, MongoDB, Git, Docker, Data Structures",
    "Projects", "Campus Queue App | React, Node.js, MongoDB",
    "- Built a queue management web app used by 200 students, reducing waiting time by 40%",
    "- Designed REST APIs and deployed on Vercel handling 1500 requests per day",
    "Log Anomaly Detector | Python, scikit-learn",
    "- Developed an isolation-forest model that cut false alerts by 35%",
    "Experience", "Software Intern, Acme Technologies Pvt Ltd, Jun 2026 - Aug 2026",
    "- Improved API latency by 30% by adding Redis caching",
    "Certifications", "- NPTEL Programming in Java",
    "Achievements", "- Won second place among 30 teams at a college hackathon",
]
JD = ("We are hiring a backend engineer. Required: Python, SQL, Docker, REST APIs, Kubernetes and strong "
      "data structures. Nice to have: React, AWS and system design experience.")


def make_docx(lines=RESUME_LINES) -> bytes:
    d = docx.Document()
    for ln in lines:
        d.add_paragraph(ln)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def make_pdf(lines=RESUME_LINES) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for ln in lines:
        c.drawString(40, y, ln[:110])
        y -= 16
    c.save()
    return buf.getvalue()


def upload(client, auth, data, name="resume.docx"):
    return client.post("/resume/upload", headers=auth, files={"file": (name, data, "application/octet-stream")})


# ------------------------------------------------------------------ auth ----
def test_signup_login_profile_and_validation(client):
    h = signup(client)
    assert client.post("/auth/signup", json={"email": "user@example.com", "name": "x", "password": "password123"}).status_code == 409
    assert client.post("/auth/signup", json={"email": "bad", "name": "x", "password": "password123"}).status_code == 422
    assert client.post("/auth/signup", json={"email": "a@b.com", "name": "x", "password": "short"}).status_code == 422
    assert client.post("/auth/login", json={"email": "user@example.com", "password": "wrong-pass"}).status_code == 401
    assert client.post("/auth/login", json={"email": "nobody@example.com", "password": "password123"}).status_code == 401
    assert client.post("/auth/login", json={"email": "user@example.com", "password": "password123"}).status_code == 200
    r = client.put("/auth/profile", headers=h, json={"target_role": "ML Engineer", "preferred_language": "ta", "company_type": "startup"})
    assert r.json()["target_role"] == "ML Engineer" and r.json()["preferred_language"] == "ta"
    assert client.put("/auth/profile", headers=h, json={"preferred_language": "fr"}).status_code == 422
    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_passwords_are_bcrypt_hashed(client):
    signup(client)
    with SessionLocal() as db:
        u = db.scalar(select(M.User))
        assert u.password_hash.startswith("$2") and "password123" not in u.password_hash


def test_admin_role_only_for_configured_emails(client):
    signup(client, "admin@example.com")
    signup(client, "other@example.com")
    with SessionLocal() as db:
        roles = {u.email: u.role for u in db.scalars(select(M.User))}
    assert roles == {"admin@example.com": "admin", "other@example.com": "user"}


def test_consent_gate_and_toggle(client):
    h = signup(client, consent=False)
    r = upload(client, h, make_docx())
    assert r.status_code == 403 and "consent" in r.json()["detail"].lower()
    assert client.post("/auth/consent", headers=h, json={"consent": True}).json()["consent_given"] is True
    assert upload(client, h, make_docx()).status_code == 200


def test_delete_all_my_data_cascades(client, auth):
    upload(client, auth, make_docx())
    q = client.post("/dsa/question", headers=auth, json={"topic": "arrays", "difficulty": "easy"}).json()
    client.post("/roadmap/generate", headers=auth, json={"timeline_months": 3, "hours_per_week": 5})
    assert client.request("DELETE", "/auth/me", headers=auth, json={"confirm": False}).status_code == 400
    exported = client.get("/auth/export", headers=auth).json()
    assert exported["resumes"] and exported["answers"] and exported["roadmaps"]
    assert client.request("DELETE", "/auth/me", headers=auth, json={"confirm": True}).json() == {"deleted": True}
    with SessionLocal() as db:
        for model in (M.User, M.Resume, M.InterviewSession, M.Answer, M.Roadmap):
            assert db.scalar(select(model.id)) is None, model.__name__
    assert client.get("/auth/me", headers=auth).status_code == 401
    assert q["answer_id"]


# ----------------------------------------------------------------- resume ----
def test_resume_docx_and_pdf_pipeline(client, auth):
    for data, name in ((make_docx(), "r.docx"), (make_pdf(), "r.pdf")):
        r = upload(client, auth, data, name)
        assert r.status_code == 200, r.text
        j = r.json()
        skills = {s.lower() for s in j["parsed"]["skills"]}
        assert {"python", "sql", "react"} <= skills
        ats = j["ats"]
        assert 0 <= ats["total"] <= 100
        assert ats["total"] == pytest.approx(sum(c["score"] for c in ats["components"].values()), abs=0.2)
        assert {c["max"] for c in ats["components"].values()} == {30, 25, 20}
        assert sum(c["max"] for c in ats["components"].values()) == 100
        assert j["suggestions"]["suggestions"]
    assert client.get("/resume/latest", headers=auth).json()["filename"] == "r.pdf"


def test_resume_errors_are_friendly(client, auth):
    assert upload(client, auth, b"", "empty.pdf").status_code == 422
    assert upload(client, auth, b"hi", "tiny.docx").status_code == 422          # not a real docx
    assert upload(client, auth, make_docx(["Only a few words"]), "short.docx").status_code == 422
    assert upload(client, auth, b"MZ binary", "evil.exe").status_code == 422
    r = upload(client, auth, make_pdf([" "]), "blank.pdf")
    assert r.status_code == 422 and "readable text" in r.json()["detail"]


def test_ats_rewards_metrics_and_structure(client, auth):
    strong = upload(client, auth, make_docx(), "a.docx").json()["ats"]["total"]
    weak = upload(client, auth, make_docx(["Rahul", "I like computers. " * 30, "Skills", "word"]), "b.docx")
    weak_total = weak.json()["ats"]["total"] if weak.status_code == 200 else 0
    assert strong > weak_total + 20


def test_resume_text_is_not_logged(client, auth, caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        upload(client, auth, make_docx())
    assert "asha.k@example.com" not in caplog.text and "Campus Queue App" not in caplog.text


def test_jd_match_rewriter_cover_letter(client, auth):
    assert client.post("/resume/jd-match", headers=auth, json={"jd_text": JD}).status_code == 404  # no resume yet
    upload(client, auth, make_docx())
    m = client.post("/resume/jd-match", headers=auth, json={"jd_text": JD}).json()
    assert {"Python", "SQL"} <= set(m["matched_skills"]) and "Kubernetes" in m["missing_skills"]
    assert 0 < m["match_pct"] < 100 and m["reasoning"]["summary"] and m["embedding"]["embedder"]
    assert client.post("/resume/jd-match", headers=auth, json={"jd_text": "Looking for a friendly person to join our team today."}).status_code == 422
    rw = client.post("/resume/rewrite-bullets", headers=auth, json={"bullets": ["Worked on a website for college fest", "Helped fix bugs"]}).json()
    assert len(rw["items"]) == 2 and rw["items"][0]["rewritten"] != rw["items"][0]["original"]
    cl = client.post("/resume/cover-letter", headers=auth, json={"jd_text": JD, "company": "Acme", "role": "Backend Intern"}).json()
    assert "Acme" in cl["body"] and cl["subject"]


# -------------------------------------------------------------------- DSA ----
def reference_for(title):
    q = next(x for x in R.load_question_bank() if x["title"] == title)
    return q["reference_solution"]


def test_dsa_full_flow_with_hints_and_adaptive_difficulty(client, auth):
    gen = client.post("/dsa/question", headers=auth, json={"topic": "arrays", "difficulty": "easy"}).json()
    q, aid = gen["question"], gen["answer_id"]
    assert "reference_solution" not in q and "test_cases" not in q and len(q["visible_tests"]) == 2
    assert q["hidden_test_count"] >= 4 and gen["generation"]["guard"]["ok"]
    ref = reference_for(q["title"])

    run = client.post(f"/dsa/{aid}/run", headers=auth, json={"language": "python", "code": ref}).json()
    assert run["summary"]["passed"] == run["summary"]["total"] == 2
    bad = client.post(f"/dsa/{aid}/run", headers=auth, json={"language": "python", "code": "print(0)"}).json()
    assert bad["summary"]["passed"] == 0

    h1 = client.post(f"/dsa/{aid}/hint", headers=auth).json()
    assert h1["hint_number"] == 1 and h1["hints_remaining"] == 2
    sub = client.post(f"/dsa/{aid}/submit", headers=auth, json={"language": "python", "code": ref}).json()
    assert sub["tests"]["passed"] == sub["tests"]["total"] and sub["tests"]["hidden_total"] >= 4
    assert sub["score"]["hint_penalty"] == 5 and sub["score"]["components"]["tests"] == 60.0
    assert sub["score"]["total"] >= 80
    assert all(c["input"] == "(hidden)" for c in sub["tests"]["cases"] if c["hidden"])
    assert sub["review"]["time_complexity"] and sub["next_difficulty"] in ("easy", "medium")

    # Adaptive: the next question in the same session is chosen from the previous score (>= 80 -> up)
    nxt = client.post("/dsa/question", headers=auth, json={"session_id": gen["session_id"], "topic": "strings"}).json()
    assert nxt["difficulty"] == "medium" and "previous score" in nxt["difficulty_reason"]


def test_dsa_wrong_code_scores_low_and_hints_cap(client, auth):
    gen = client.post("/dsa/question", headers=auth, json={"topic": "graphs", "difficulty": "medium"}).json()
    aid = gen["answer_id"]
    sub = client.post(f"/dsa/{aid}/submit", headers=auth, json={"language": "python", "code": "print('WRONG_ANSWER_TOKEN')"}).json()
    assert sub["tests"]["passed"] == 0 and sub["score"]["components"]["tests"] == 0
    assert sub["score"]["total"] <= 14
    for _ in range(3):
        assert client.post(f"/dsa/{aid}/hint", headers=auth).status_code == 200
    assert client.post(f"/dsa/{aid}/hint", headers=auth).status_code == 409


def test_dsa_timeout_unsupported_language_and_sql(client, auth):
    aid = client.post("/dsa/question", headers=auth, json={"topic": "arrays", "difficulty": "easy"}).json()["answer_id"]
    t = client.post(f"/dsa/{aid}/run", headers=auth, json={"language": "python", "code": "while True: pass"}).json()
    assert {c["status"] for c in t["summary"]["cases"]} == {"timeout"}
    assert client.post(f"/dsa/{aid}/run", headers=auth, json={"language": "ruby", "code": "puts 1"}).status_code == 422

    sq = client.post("/dsa/question", headers=auth, json={"topic": "sql", "difficulty": "easy"}).json()
    ref = reference_for(sq["question"]["title"])
    sub = client.post(f"/dsa/{sq['answer_id']}/submit", headers=auth, json={"language": "python", "code": ref}).json()
    assert sub["tests"]["passed"] == sub["tests"]["total"]  # SQL questions always run as SQL


def test_dsa_isolation_between_users(client, auth):
    aid = client.post("/dsa/question", headers=auth, json={"topic": "arrays", "difficulty": "easy"}).json()["answer_id"]
    other = signup(client, "other@example.com")
    assert client.post(f"/dsa/{aid}/hint", headers=other).status_code == 404
    assert client.post(f"/dsa/{aid}/submit", headers=other, json={"language": "python", "code": "print(1)"}).status_code == 404


# ------------------------------------------------------------ soft skills ----
GOOD_ANSWER = ("In my final-year project two teammates disagreed on the database and it blocked us for a week. As team lead "
               "I set up a benchmark of both options, we reviewed the numbers together and chose MySQL. We delivered four "
               "days early with 30 percent fewer bugs, and I learned to move debates to data.")


def test_soft_flow_dual_evaluation_and_followups(client, auth):
    upload(client, auth, make_docx())
    qs = client.post("/soft/questions", headers=auth, json={"count": 3}).json()
    assert len(qs["questions"]) == 3 and {q["category"] for q in qs["questions"]} <= {"behavioral", "situational", "hr", "resume"}
    first = qs["questions"][0]["answer_id"]

    # strong answer: both evaluators, analytics, and (almost certainly) no follow-up needed
    r = client.post(f"/soft/{first}/answer", headers=auth, json={"text": GOOD_ANSWER, "duration_sec": 40}).json()
    rub = r["rubric"]
    for k in ("clarity", "structure", "relevance", "confidence", "grammar_vocabulary", "depth"):
        assert 0 <= rub[k]["score"] <= 10 and rub[k]["reason"]
    assert r["classifier"]["label"] in ("poor", "average", "good") and len(r["classifier"]["probabilities"]) == 3
    assert r["analytics"]["wpm"] and r["analytics"]["source"] == "voice" and r["analytics"]["sentiment"]["label"]
    assert 0 <= r["score"]["total"] <= 100 and set(r["score"]["weights"]) == {"rubric", "classifier", "delivery"}
    assert client.post(f"/soft/{first}/answer", headers=auth, json={"text": "again"}).status_code == 409

    # weak answer -> follow-up chain, max 2 follow-ups, then it stops
    second = qs["questions"][1]["answer_id"]
    cur, depths = second, []
    for _ in range(4):
        out = client.post(f"/soft/{cur}/answer", headers=auth, json={"text": "Um, I think it was fine, basically."}).json()
        if not out["followup"]:
            break
        depths.append(out["followup"]["depth"])
        cur = out["followup"]["answer_id"]
    assert depths == [1, 2]


def test_soft_validation_and_voice_fallback(client, auth):
    qs = client.post("/soft/questions", headers=auth, json={"count": 1}).json()
    aid = qs["questions"][0]["answer_id"]
    assert client.post(f"/soft/{aid}/answer", headers=auth, json={"text": "   "}).status_code == 422
    assert client.post(f"/soft/{aid}/answer", headers=auth, json={"text": ""}).status_code == 422
    # Empty audio -> friendly 503 telling the user to check mic permission / type instead
    r = client.post("/soft/transcribe", headers=auth, files={"audio": ("a.wav", b"", "audio/wav")})
    assert r.status_code == 503 and "microphone" in r.json()["detail"].lower()
    r = client.post("/soft/transcribe", headers=auth, files={"audio": ("a.xyz", b"123", "audio/wav")})
    assert r.status_code == 503 and "unsupported" in r.json()["detail"].lower()


def test_speech_analytics_numbers():
    from backend.services.nlp_service import speech_analytics

    a = speech_analytics("Um I basically worked on like a project and uh finished it", duration_sec=6,
                         segments=[{"start": 0, "end": 2}, {"start": 3.5, "end": 6}])
    assert a["filler_count"] >= 4 and a["wpm"] == pytest.approx(12 / 6 * 60, abs=0.2)
    assert a["pauses"]["count"] == 1 and a["pauses"]["longest_sec"] == 1.5 and a["pace"] == "good pace"  # 120 wpm


# ----------------------------------------------- mock interview + feedback ----
def test_mock_interview_timer_is_enforced(client, auth):
    s = client.post("/sessions", headers=auth, json={"kind": "mock", "time_limit_min": 10}).json()
    assert s["seconds_remaining"] > 500
    q = client.post("/dsa/question", headers=auth, json={"session_id": s["id"], "topic": "arrays", "difficulty": "easy"})
    assert q.status_code == 200
    with SessionLocal() as db:  # pretend the session started 30 minutes ago
        row = db.get(M.InterviewSession, s["id"])
        row.started_at = row.started_at - timedelta(minutes=30)
        db.commit()
    late = client.post(f"/dsa/{q.json()['answer_id']}/submit", headers=auth, json={"language": "python", "code": "print(1)"})
    assert late.status_code == 409 and "time is up" in late.json()["detail"].lower()
    assert client.get(f"/sessions/{s['id']}", headers=auth).json()["seconds_remaining"] == 0


def run_mock_session(client, auth):
    s = client.post("/sessions", headers=auth, json={"kind": "mock", "time_limit_min": 60}).json()
    sid = s["id"]
    g = client.post("/dsa/question", headers=auth, json={"session_id": sid, "topic": "trees", "difficulty": "easy"}).json()
    client.post(f"/dsa/{g['answer_id']}/submit", headers=auth, json={"language": "python", "code": "print(0)"})
    sq = client.post("/soft/questions", headers=auth, json={"session_id": sid, "count": 1}).json()
    client.post(f"/soft/{sq['questions'][0]['answer_id']}/answer", headers=auth, json={"text": "Um, I think it was fine, basically."})
    return sid


def test_feedback_chain_dashboard_pdf_and_languages(client, auth):
    sid = run_mock_session(client, auth)
    fb = client.post(f"/feedback/{sid}", headers=auth, json={}).json()["content"]
    for key in ("summary", "strengths", "weaknesses", "skill_gaps", "per_question", "suggestions", "explanations", "disclaimer"):
        assert fb[key], key
    assert "guidance only" in fb["disclaimer"].lower() and fb["meta"]["chain"][0] == "feedback_summary"
    tax = R.load_taxonomy()
    assert fb["skill_gaps"] and all(g["skill_id"] in tax for g in fb["skill_gaps"])
    assert any(q["model_answer"] for q in fb["per_question"] if q["type"] == "soft")
    assert any(e["rubric"] for e in fb["explanations"] if e["type"] == "soft")   # rubric reasons panel data

    stored = client.get(f"/feedback/{sid}", headers=auth).json()
    assert stored["language"] == "en" and stored["content_en"]["summary"] == fb["summary"]
    ta = client.post(f"/feedback/{sid}/translate", headers=auth, json={"language": "ta"}).json()
    assert ta["language"] == "ta" and ta["content"]["meta"]["translation"]["ok"] is False   # mock cannot translate
    assert "unavailable" in ta["content"]["meta"]["translation"]["note"].lower()

    pdf = client.get(f"/feedback/{sid}/report.pdf", headers=auth)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF") and len(pdf.content) > 2000

    dash = client.get("/dashboard", headers=auth).json()
    assert dash["readiness"]["score"] is not None and dash["trend"] and dash["radar"]
    assert client.get(f"/feedback/{sid}", headers=signup(client, "x@example.com")).status_code == 404


def test_feedback_requires_scored_answers(client, auth):
    sid = client.post("/sessions", headers=auth, json={"kind": "dsa"}).json()["id"]
    assert client.post(f"/feedback/{sid}", headers=auth, json={}).status_code == 422


# ---------------------------------------------------------------- roadmap ----
def test_roadmap_progress_and_regeneration_uses_latest_gaps(client, auth):
    rm = client.post("/roadmap/generate", headers=auth, json={"timeline_months": 3, "hours_per_week": 6}).json()
    catalog = R.course_ids()
    assert rm["content"]["recommended_courses"] and all(c["course_id"] in catalog for c in rm["content"]["recommended_courses"])
    assert rm["progress"]["percent"] == 0 and rm["progress"]["total"] > 5
    key = "w1-t0"
    p = client.post(f"/roadmap/{rm['id']}/progress", headers=auth, json={"item_key": key, "done": True}).json()
    assert p["completed"] == 1 and p["percent"] > 0
    assert client.post(f"/roadmap/{rm['id']}/progress", headers=auth, json={"item_key": "bogus", "done": True}).status_code == 422
    p = client.post(f"/roadmap/{rm['id']}/progress", headers=auth, json={"item_key": key, "done": False}).json()
    assert p["completed"] == 0

    sid = run_mock_session(client, auth)       # weak trees answer + weak soft answer -> new skill gaps
    client.post(f"/feedback/{sid}", headers=auth, json={})
    new = client.post(f"/roadmap/{rm['id']}/regenerate", headers=auth).json()
    assert new["id"] != rm["id"] and "trees" in new["skill_gaps"]
    assert client.get("/roadmap/active", headers=auth).json()["id"] == new["id"]
    assert client.post("/roadmap/generate", headers=auth, json={"timeline_months": 5, "hours_per_week": 6}).status_code == 422


# ---------------------------------------------------------------- prompts ----
def test_prompt_playground(client, auth):
    lst = client.get("/prompts", headers=auth).json()
    names = {p["name"] for p in lst["prompts"]}
    assert {"rubric_evaluation", "dsa_question_generation", "roadmap_generation"} <= names and "RubricEvaluation" in lst["schemas"]
    t = client.get("/prompts/rubric_evaluation", headers=auth).json()
    assert t["version"] == 2 and t["sample_variables"]["answer"]
    assert client.get("/prompts/rubric_evaluation?version=1", headers=auth).json()["version"] == 1
    assert client.get("/prompts/nope", headers=auth).status_code == 404

    body = {"system": t["system"], "user_template": t["user"], "variables": t["sample_variables"], "temperature": 0.2,
            "provider": "mock", "prompt_name": "rubric_evaluation"}
    out = client.post("/prompts/run", headers=auth, json=body).json()
    assert out["json_valid"] and out["schema_valid"] is True and out["schema"] == "RubricEvaluation" and out["provider"] == "mock"
    bad = client.post("/prompts/run", headers=auth, json={**body, "user_template": "{{ missing_var }}"})
    assert bad.status_code == 422 and "Template error" in bad.json()["detail"]
    notjson = client.post("/prompts/run", headers=auth, json={**body, "prompt_name": None, "schema_name": "CoverLetter",
                                                              "user_template": "hello {{ question }}"}).json()
    assert notjson["schema_valid"] is False or notjson["json_valid"] is False

    v1 = client.post("/prompts/versions", headers=auth, json={"name": "rubric_evaluation", "version_label": "strict-v1",
                                                              "system": "S1", "user_template": "U1 {{ question }}", "temperature": 0.1}).json()
    v2 = client.post("/prompts/versions", headers=auth, json={"name": "rubric_evaluation", "version_label": "strict-v2",
                                                              "system": "S2", "user_template": "U2 {{ answer }}", "temperature": 0.3}).json()
    assert client.get(f"/prompts/saved/{v1['id']}", headers=auth).json()["system"] == "S1"
    assert client.get(f"/prompts/saved/{v2['id']}", headers=auth).json()["temperature"] == 0.3
    listed = next(p for p in client.get("/prompts", headers=auth).json()["prompts"] if p["name"] == "rubric_evaluation")
    assert [s["label"] for s in listed["saved_versions"]] == ["strict-v1", "strict-v2"]
    assert client.post("/prompts/versions", headers=auth, json={"name": "unknown_prompt", "version_label": "x", "system": "", "user_template": "u"}).status_code == 422


def test_playground_is_rate_limited(client, auth):
    body = {"system": "s", "user_template": "hi", "provider": "mock"}
    codes = [client.post("/prompts/run", headers=auth, json=body).status_code for _ in range(22)]
    assert codes[:20] == [200] * 20 and codes[20:] == [429, 429]


# ------------------------------------------------------------------ admin ----
def test_admin_dashboard_is_protected_and_anonymised(client):
    user = signup(client, "stud1@example.com", name="Secret Student")
    run_mock_session(client, user)
    sid = client.get("/sessions", headers=user).json()[0]["id"]
    client.post(f"/feedback/{sid}", headers=user, json={})
    assert client.get("/admin/overview", headers=user).status_code == 403
    assert client.get("/admin/overview").status_code == 401

    admin = signup(client, "admin@example.com", name="Placement Cell")
    ov = client.get("/admin/overview", headers=admin).json()
    assert ov["batch"]["consenting_users"] == 1 and ov["distributions"]["readiness"] and ov["common_skill_gaps"]
    assert ov["privacy"]["anonymized"] is True
    csv = client.get("/admin/export.csv", headers=admin)
    assert csv.status_code == 200 and csv.text.startswith("student,sessions")
    assert "student-" in csv.text and "stud1" not in csv.text and "Secret" not in csv.text


def test_admin_small_batch_suppression(client, monkeypatch):
    from backend.core.config import get_settings

    monkeypatch.setenv("ADMIN_MIN_GROUP", "5")
    get_settings.cache_clear()
    user = signup(client, "stud1@example.com")
    run_mock_session(client, user)
    admin = signup(client, "admin@example.com")
    ov = client.get("/admin/overview", headers=admin).json()
    assert ov["privacy"]["per_student_views_suppressed"] is True and ov["top_improvers"] == []
    assert client.get("/admin/export.csv", headers=admin).status_code == 409


def test_users_who_withheld_consent_are_excluded_from_admin(client):
    signup(client, "nocon@example.com", consent=False)
    admin = signup(client, "admin@example.com")
    assert client.get("/admin/overview", headers=admin).json()["batch"]["consenting_users"] == 0


def test_health_reports_capabilities(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm"]["provider"] == "mock" and h["llm"]["ready"]
    assert h["sandbox"]["languages"]["python"] and h["embeddings"]["embedder"] == "hash-bow-512"
