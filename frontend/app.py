"""PrepWise AI - Home: login / sign-up, profile, consent, data export and deletion.

Run:  streamlit run frontend/app.py
"""
import json

import api  # noqa: E402  (frontend/ is on sys.path when run via `streamlit run frontend/app.py`)
import streamlit as st

st.set_page_config(page_title="PrepWise AI", page_icon="🎯", layout="wide")
api.sidebar_status()

st.title("🎯 PrepWise AI")
st.caption("Personalized interview preparation and career guidance - fixed AI pipelines, RAG over a local question "
           "bank, sandboxed code execution and explainable scores. No autonomous agents.")

if not st.session_state.get("token"):
    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
    with login_tab, st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            res = api.try_call(api.post, "/auth/login", {"email": email, "password": password})
            if res:
                st.session_state.update(token=res["token"], user=res["user"])
                st.rerun()
    with signup_tab, st.form("signup"):
        name = st.text_input("Name")
        email = st.text_input("Email ", key="su_email")
        password = st.text_input("Password (min 8 characters)", type="password", key="su_pw")
        consent = st.checkbox("I consent to PrepWise AI processing my resume and practice answers to give feedback. "
                              "I can withdraw consent or delete all my data at any time.")
        if st.form_submit_button("Create account", type="primary"):
            res = api.try_call(api.post, "/auth/signup", {"name": name, "email": email, "password": password,
                                                           "consent": consent})
            if res:
                st.session_state.update(token=res["token"], user=res["user"])
                st.rerun()
    st.markdown("""
**What you can do here**
1. Upload a resume → ATS score, JD match, bullet rewriter, cover letter
2. Practise DSA with a code editor, sandboxed tests, hints and adaptive difficulty
3. Practise HR/behavioural answers by text or voice with rubric + classifier scoring
4. Run a timed mock interview and get a multilingual feedback report (English / தமிழ் / हिन्दी)
5. Track an Interview Readiness Score and follow a course-backed roadmap
""")
    st.stop()

user = api.refresh_user()
if not user:
    st.stop()

st.subheader(f"Welcome, {user['name']}")
col1, col2 = st.columns(2)
with col1, st.form("profile"):
    st.markdown("**Profile**")
    name = st.text_input("Name", user["name"])
    role = st.text_input("Target role", user.get("target_role") or "Software Engineer")
    ctype = st.selectbox("Target company type", ["service", "product", "startup"],
                         index=["service", "product", "startup"].index(user.get("company_type") or "product"))
    level = st.selectbox("Experience", ["fresher", "junior", "mid"],
                         index=["fresher", "junior", "mid"].index(user.get("experience_level") or "fresher"))
    lang = st.selectbox("Feedback language", ["en", "ta", "hi"], index=["en", "ta", "hi"].index(user.get("preferred_language") or "en"),
                        format_func={"en": "English", "ta": "தமிழ் (Tamil)", "hi": "हिन्दी (Hindi)"}.get)
    if st.form_submit_button("Save profile"):
        if api.try_call(api.call, "PUT", "/auth/profile", json={
                "name": name, "target_role": role, "company_type": ctype, "experience_level": level,
                "preferred_language": lang}):
            api.refresh_user()
            st.success("Profile saved.")
with col2:
    st.markdown("**Privacy & data**")
    consent = st.toggle("Consent to data processing", value=user["consent_given"],
                        help="Without consent AI features are disabled and your data is excluded from admin statistics.")
    if consent != user["consent_given"] and api.try_call(api.post, "/auth/consent", {"consent": consent}):
        api.refresh_user()
        st.rerun()
    st.caption("Raw resume text is never written to logs. The admin dashboard only sees anonymised aggregates.")
    export = api.try_call(api.get, "/auth/export")
    if export:
        st.download_button("⬇️ Export all my data (JSON)", json.dumps(export, indent=2, default=str),
                           "prepwise-my-data.json", "application/json")
    with st.expander("🗑️ Delete my account and all data"):
        st.warning("This permanently deletes your profile, resumes, answers, feedback and roadmaps.")
        if st.checkbox("I understand this cannot be undone") and st.button("Delete everything", type="primary"):
            if api.try_call(api.call, "DELETE", "/auth/me", json={"confirm": True}) is not None:
                st.session_state.clear()
                st.success("All data deleted.")
                st.rerun()
api.disclaimer()
