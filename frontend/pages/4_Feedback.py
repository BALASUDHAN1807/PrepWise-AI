"""Timed mock interview (DSA round + HR round) and the post-session feedback report (multilingual, PDF)."""
import time

import api
import streamlit as st
from components import soft_answer_block
from streamlit_ace import st_ace

user = api.require_login()
st.title("🎤 Feedback & Mock Interview")

LANG_NAMES = {"en": "English", "ta": "தமிழ் (Tamil)", "hi": "हिन्दी (Hindi)"}
TEMPLATE = "import sys\n\n\ndef main():\n    data = sys.stdin.read().split()\n    print(0)\n\n\nmain()\n"
tab_mock, tab_fb = st.tabs(["Run a mock interview", "Feedback reports"])


def render_feedback(c: dict, session_id: int) -> None:
    st.subheader(f"Overall score: {c.get('overall_score', 0):.0f}/100")
    st.write(c["summary"])
    a, b = st.columns(2)
    with a:
        st.markdown("**Strengths**")
        for x in c.get("strengths", []):
            st.markdown(f"- ✅ {x}")
    with b:
        st.markdown("**Weaknesses**")
        for x in c.get("weaknesses", []):
            st.markdown(f"- ⚠️ {x}")
    if c.get("skill_gaps"):
        st.markdown("**Skill gaps (validated against the skills taxonomy)**")
        icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        for g in c["skill_gaps"]:
            st.markdown(f"- {icon.get(g['severity'], '•')} **{g.get('name', g['skill_id'])}** - {g['reason']}")
    st.markdown("**Per-question feedback & model answers**")
    for pq in c.get("per_question", []):
        with st.expander(f"{pq['question'][:90]} - {pq['score']:.0f}/100"):
            st.write(pq["feedback"])
            if pq.get("model_answer"):
                st.markdown("**Example of a stronger answer**")
                st.info(pq["model_answer"])
    st.markdown("**Improvement plan**")
    for s in c.get("suggestions", []):
        st.markdown(f"- {s}")
    with st.expander("🔍 Why did I get this score? (explanations panel)"):
        for e in c.get("explanations", []):
            st.markdown(f"**{e['question'][:100]}** ({e['type']}) → {e['score']:.0f}/100")
            st.json(e.get("breakdown", {}), expanded=False)
            if e.get("rubric"):
                st.json(e["rubric"], expanded=False)
        meta = c.get("meta", {})
        st.caption(f"Prompt chain executed (fixed order): {' → '.join(meta.get('chain', []))}. "
                   f"Skill ids rejected by validation: {meta.get('rejected_skill_ids') or 'none'}.")
    st.caption(c.get("disclaimer", api.DISCLAIMER))
    pdf = api.try_call(api.get, f"/feedback/{session_id}/report.pdf", raw=True)
    if pdf:
        st.download_button("⬇️ Download PDF report", pdf, f"prepwise-report-{session_id}.pdf", "application/pdf")


# ------------------------------------------------------------------ mock run --
with tab_mock:
    m = st.session_state.get("mock")
    if not m:
        c1, c2, c3 = st.columns(3)
        minutes = c1.number_input("Time limit (minutes)", 10, 120, 45, step=5)
        n_dsa = c2.number_input("DSA questions", 1, 3, 1)
        n_hr = c3.number_input("HR questions", 1, 5, 2)
        st.caption("The time limit is enforced by the server: submissions after time is up are rejected.")
        if st.button("▶ Start mock interview", type="primary"):
            s = api.try_call(api.post, "/sessions", {"kind": "mock", "time_limit_min": int(minutes),
                                                      "dsa_count": int(n_dsa), "hr_count": int(n_hr)})
            if s:
                st.session_state["mock"] = {"id": s["id"], "deadline": time.time() + s["seconds_remaining"],
                                            "n_dsa": int(n_dsa), "n_hr": int(n_hr), "dsa": [], "hr": None, "feedback": None}
                st.rerun()
    else:
        @st.fragment(run_every=1)
        def timer():
            left = max(0, int(m["deadline"] - time.time()))
            st.metric("⏱️ Time remaining", f"{left // 60:02d}:{left % 60:02d}")
            if left == 0:
                st.error("Time is up - further submissions are rejected. Generate your feedback below.")

        timer()
        st.markdown("### Round 1 - Coding")
        if len(m["dsa"]) < m["n_dsa"] and st.button("Get next coding question"):
            topic = ["arrays", "strings", "trees", "graphs", "dp"][len(m["dsa"]) % 5]
            q = api.try_call(api.post, "/dsa/question", {"session_id": m["id"], "topic": topic},
                             spinner="Generating and verifying question...")
            if q:
                m["dsa"].append({"q": q, "result": None})
                st.rerun()
        for idx, item in enumerate(m["dsa"]):
            q, aid = item["q"]["question"], item["q"]["answer_id"]
            with st.container(border=True):
                st.markdown(f"**{idx + 1}. {q['title']}** · {item['q']['difficulty']}")
                st.markdown(q["statement"])
                st.caption(f"Input: {q['input_format']} | Output: {q['output_format']}")
                for ex in q.get("examples", [])[:1]:
                    st.code(f"{ex['input']}\n→ {ex['output']}", language="text")
                if item["result"]:
                    r = item["result"]
                    st.success(f"Score {r['score']['total']:.0f}/100 · {r['tests']['passed']}/{r['tests']['total']} tests passed")
                    continue
                is_sql = q.get("kind") == "sql"
                code = st_ace(value="SELECT 1;" if is_sql else TEMPLATE, language="sql" if is_sql else "python", theme="monokai",
                              height=280, key=f"mock_ace_{aid}", auto_update=True)
                if st.button("Submit solution", key=f"mock_sub_{aid}", type="primary"):
                    res = api.try_call(api.post, f"/dsa/{aid}/submit", {"language": "sql" if is_sql else "python", "code": code},
                                       spinner="Running tests...")
                    if res:
                        item["result"] = res
                        st.rerun()

        st.markdown("### Round 2 - HR / Behavioural")
        if m["hr"] is None:
            if st.button("Start HR round"):
                res = api.try_call(api.post, "/soft/questions", {"session_id": m["id"], "count": m["n_hr"]},
                                   spinner="Preparing questions...")
                if res:
                    m["hr"] = [{**q, "result": None, "followup": False} for q in res["questions"]]
                    st.rerun()
        else:
            hr = m["hr"]
            i = 0
            while i < len(hr):
                it = hr[i]
                with st.container(border=True):
                    res = soft_answer_block(it, key=f"mock{it['answer_id']}")
                    if res:
                        it["result"] = res
                        if res.get("followup"):
                            fu = res["followup"]
                            hr.insert(i + 1, {"answer_id": fu["answer_id"], "question": fu["question"],
                                              "category": f"follow-up {fu['depth']}/{fu['max']}", "result": None, "followup": True})
                        st.rerun()
                i += 1

        st.markdown("### Finish")
        lang = st.selectbox("Report language", list(LANG_NAMES), format_func=LANG_NAMES.get,
                            index=list(LANG_NAMES).index(user.get("preferred_language") or "en"))
        if st.button("📝 End interview & generate feedback", type="primary"):
            fb = api.try_call(api.post, f"/feedback/{m['id']}", {"language": lang},
                              spinner="Running the feedback chain (summary → skill gaps → per-question → plan)...")
            if fb:
                m["feedback"] = fb["content"]
        if m.get("feedback"):
            render_feedback(m["feedback"], m["id"])
        if st.button("Discard and start over"):
            st.session_state.pop("mock")
            st.rerun()

# ------------------------------------------------------------- past reports --
with tab_fb:
    sessions = api.try_call(api.get, "/sessions") or []
    if not sessions:
        st.info("No sessions yet. Practise DSA or soft skills, or run a mock interview.")
    else:
        pick = st.selectbox("Session", sessions, format_func=lambda s: f"#{s['id']} · {s['kind']} · "
                            f"{s['started_at'][:16].replace('T', ' ')} · score {s['overall_score'] if s['overall_score'] is not None else '-'}"
                            f"{' · report ✓' if s['has_feedback'] else ''}")
        c1, c2 = st.columns([1, 2])
        lang = c1.selectbox("Language", list(LANG_NAMES), format_func=LANG_NAMES.get, key="fb_lang")
        if c2.button("Generate / translate report", type="primary"):
            if pick["has_feedback"]:
                fb = api.try_call(api.post, f"/feedback/{pick['id']}/translate", {"language": lang}, spinner="Translating...")
            else:
                fb = api.try_call(api.post, f"/feedback/{pick['id']}", {"language": lang}, spinner="Generating feedback...")
            if fb:
                st.session_state["fb_view"] = (pick["id"], fb["content"])
                st.rerun()
        elif pick["has_feedback"] and st.session_state.get("fb_view", (None,))[0] != pick["id"]:
            fb = api.try_call(api.get, f"/feedback/{pick['id']}")
            if fb:
                st.session_state["fb_view"] = (pick["id"], fb["content"])
        view = st.session_state.get("fb_view")
        if view and view[0] == pick["id"]:
            if lang != "en":
                st.caption("Machine-translated by the LLM; scores and skill ids are unchanged. Quality for Tamil/Hindi "
                           "should be reviewed by a native speaker.")
            render_feedback(view[1], view[0])
