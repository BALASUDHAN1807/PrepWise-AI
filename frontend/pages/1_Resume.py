"""Resume Analyzer: parsing, ATS score with breakdown, JD match, bullet rewriter, cover letter."""
import api
import plotly.graph_objects as go
import streamlit as st

user = api.require_login()
st.title("📄 Resume Analyzer")
st.caption("Parsing uses pdfplumber / python-docx + spaCy; the ATS score is a transparent rule-based formula; "
           "the LLM only writes the explanations and rewrites.")


def show_result(res: dict) -> None:
    ats = res["ats"]
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=ats["total"], title={"text": "ATS score"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"}}))
        fig.update_layout(height=260, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Method: {ats.get('method')} · {ats.get('word_count')} words")
    with c2:
        st.markdown("**Score breakdown (explainable)**")
        for name, comp in ats["components"].items():
            st.progress(min(1.0, comp["score"] / comp["max"]), text=f"{name.title()}: {comp['score']:.0f} / {comp['max']}")
            for d in comp.get("details", []):
                st.caption(f"• {d}")
    sugg = res.get("suggestions") or {}
    if sugg:
        st.markdown("**Suggestions**")
        st.write(sugg.get("summary", ""))
        for s in sugg.get("suggestions", []):
            st.markdown(f"- {s}")
        api.llm_note(sugg.get("llm"))
    parsed = res["parsed"]
    with st.expander("Parsed resume (what the system understood)", expanded=True):
        st.markdown("**Skills detected:** " + (", ".join(f"`{s}`" for s in parsed.get("skills", [])) or "none"))
        a, b = st.columns(2)
        with a:
            st.markdown("**Education**")
            for e in parsed.get("education", []):
                st.write(f"{e.get('institution', '')} {('· ' + e['year']) if e.get('year') else ''} {e.get('score', '')}")
            st.markdown("**Experience**")
            for e in parsed.get("experience", []):
                st.write(f"{e.get('company', '')} {e.get('duration', '')}")
                for h in e.get("highlights", []):
                    st.caption(f"• {h}")
        with b:
            st.markdown("**Projects**")
            for p in parsed.get("projects", []):
                st.write(f"{p.get('name', '')} - {', '.join(p.get('tech', []))}")
                st.caption(p.get("description", ""))
            for label, key in (("Certifications", "certifications"), ("Achievements", "achievements")):
                if parsed.get(key):
                    st.markdown(f"**{label}**")
                    for x in parsed[key]:
                        st.caption(f"• {x}")


tab_up, tab_jd, tab_rw, tab_cl = st.tabs(["Upload & ATS", "JD match", "Bullet rewriter", "Cover letter"])

with tab_up:
    st.caption("Your resume text is stored for your own use only and is never written to server logs. "
               "You can delete it from the Home page.")
    f = st.file_uploader("Upload resume (PDF or DOCX, max 5 MB)", type=["pdf", "docx"])
    if f and st.button("Analyze resume", type="primary"):
        res = api.try_call(api.call, "POST", "/resume/upload", files={"file": (f.name, f.getvalue(), f.type)},
                           spinner="Parsing and scoring...")
        if res:
            st.session_state["resume_result"] = res
    if "resume_result" not in st.session_state:
        try:
            latest = api.get("/resume/latest")
            st.session_state["resume_result"] = {"parsed": latest["parsed"], "ats": latest["ats"],
                                                 "suggestions": latest.get("suggestions")}
            st.caption(f"Showing your latest upload: {latest.get('filename')}")
        except api.ApiError:
            pass
    if "resume_result" in st.session_state:
        show_result(st.session_state["resume_result"])

with tab_jd:
    jd = st.text_area("Paste a job description", height=220, key="jd_text")
    if st.button("Match my resume", type="primary"):
        m = api.try_call(api.post, "/resume/jd-match", {"jd_text": jd}, spinner="Embedding and comparing...")
        if m:
            st.session_state["jd_match"] = m
    m = st.session_state.get("jd_match")
    if m:
        st.metric("Skill match", f"{m['match_pct']:.0f}%", help="Share of the JD's skills your resume covers (semantic match)")
        a, b = st.columns(2)
        a.success("Matched: " + (", ".join(m["matched_skills"]) or "-"))
        b.error("Missing: " + (", ".join(m["missing_skills"]) or "-"))
        r = m["reasoning"]
        st.write(r.get("summary", ""))
        for k, label in (("strengths", "Strengths"), ("gaps", "Gaps"), ("recommendations", "Recommendations")):
            if r.get(k):
                st.markdown(f"**{label}**")
                for x in r[k]:
                    st.markdown(f"- {x}")
        with st.expander("How the match was computed"):
            st.dataframe(m["details"], width="stretch")
            st.caption(f"Embedder: {m['embedding']['embedder']} (semantic={m['embedding'].get('semantic')}) · "
                       f"document similarity {m['document_similarity']}")
        api.llm_note(m.get("llm"))

with tab_rw:
    raw = st.text_area("One resume bullet per line (max 10)", height=160,
                       placeholder="Worked on a website for college fest\nHelped fix bugs")
    role = st.text_input("Target role (optional)", user.get("target_role") or "")
    if st.button("Rewrite bullets", type="primary"):
        bullets = [x.strip() for x in raw.splitlines() if x.strip()][:10]
        if not bullets:
            st.warning("Enter at least one bullet.")
        else:
            res = api.try_call(api.post, "/resume/rewrite-bullets", {"bullets": bullets, "target_role": role or None},
                               spinner="Rewriting...")
            if res:
                for it in res["items"]:
                    st.markdown(f"~~{it['original']}~~")
                    st.success(it["rewritten"])
                    st.caption(it.get("rationale", ""))
                st.caption("Only add numbers you can truthfully back up - the rewriter never invents metrics.")
                api.llm_note(res.get("llm"))

with tab_cl:
    c1, c2, c3 = st.columns(3)
    company = c1.text_input("Company")
    role = c2.text_input("Role", user.get("target_role") or "", key="cl_role")
    tone = c3.selectbox("Tone", ["professional", "enthusiastic", "concise"])
    jd2 = st.text_area("Job description", height=180, key="cl_jd")
    if st.button("Generate cover letter", type="primary"):
        res = api.try_call(api.post, "/resume/cover-letter",
                           {"jd_text": jd2, "company": company, "role": role, "tone": tone}, spinner="Writing...")
        if res:
            st.text_input("Subject", res["subject"])
            st.text_area("Body (edit before sending)", res["body"], height=320)
            st.download_button("Download .txt", f"Subject: {res['subject']}\n\n{res['body']}", "cover_letter.txt")
            api.llm_note(res.get("llm"))
api.disclaimer()
