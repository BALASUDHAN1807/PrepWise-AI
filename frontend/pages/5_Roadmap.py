"""Personalised learning roadmap: weekly plan, milestones, projects and courses retrieved from the local catalog."""
import api
import streamlit as st

user = api.require_login()
st.title("🗺️ Career Roadmap")
st.caption("Built from your latest skill gaps. Courses come only from the local catalog (retrieved by RAG and "
           "validated by id) - the model cannot invent links.")


def show(rm: dict) -> None:
    c = rm["content"]
    p = rm["progress"]
    st.subheader(c["title"])
    st.write(c["summary"])
    st.progress(p["percent"] / 100, text=f"Progress: {p['completed']}/{p['total']} items ({p['percent']:.0f}%)")
    done = set(p["done"])
    if rm.get("skill_gaps"):
        st.markdown("**Based on gaps:** " + ", ".join(f"`{g}`" for g in c["meta"].get("skill_gap_names", rm["skill_gaps"])))

    def check(key: str, label: str) -> None:
        val = st.checkbox(label, value=key in done, key=f"rm{rm['id']}_{key}")
        if val != (key in done):
            if api.try_call(api.post, f"/roadmap/{rm['id']}/progress", {"item_key": key, "done": val}) is not None:
                st.rerun()

    t1, t2, t3, t4 = st.tabs(["Weekly plan", "Courses", "Projects & milestones", "Phases"])
    with t1:
        for w in c["weekly_plan"]:
            with st.expander(f"Week {w['week']} - {w['focus']}"):
                for i, task in enumerate(w["tasks"]):
                    check(f"w{w['week']}-t{i}", task)
    with t2:
        if not c["recommended_courses"]:
            st.info("No catalog course matched your gaps yet.")
        for co in c["recommended_courses"]:
            with st.container(border=True):
                st.markdown(f"**[{co['title']}]({co['url']})** · {co['platform']} · {co['level']}")
                st.caption(f"{co.get('provider', '')} · for *{co.get('skill', '')}* - {co.get('why', '')}")
                check(f"course-{co['course_id']}", "Completed")
        rej = c["meta"].get("rejected_course_ids") or []
        if rej:
            st.caption(f"Guard: {len(rej)} course id(s) proposed by the model were not in the catalog and were dropped.")
    with t3:
        for pr in c["projects"]:
            st.markdown(f"**{pr['title']}** - {pr['description']}")
        st.markdown("**Milestones**")
        for i, ms in enumerate(c["milestones"]):
            check(f"ms-{i}", f"Week {ms['week']}: {ms['title']}")
    with t4:
        for ph in c["phases"]:
            st.markdown(f"**{ph['name']}** (weeks {ph['weeks']})")
            st.write("Goals: " + "; ".join(ph["goals"]))
            st.caption("Skills: " + ", ".join(ph["skills"]))
    api.llm_note(c["meta"].get("llm"))
    if st.button("🔄 Regenerate from my latest results"):
        if api.try_call(api.post, f"/roadmap/{rm['id']}/regenerate", spinner="Re-planning from your newest feedback..."):
            st.rerun()


rm = None
try:
    rm = api.get("/roadmap/active")
except api.ApiError as exc:
    if exc.status != 404:
        st.error(str(exc))
if rm:
    show(rm)
    st.divider()
    st.caption("Want a different target or timeline? Generate a new roadmap below (the old one is archived).")
with st.expander("Create a new roadmap", expanded=rm is None):
    c1, c2, c3 = st.columns(3)
    role = c1.text_input("Target role", user.get("target_role") or "Software Engineer")
    months = c2.selectbox("Timeline (months)", [3, 6, 12], index=1)
    hours = c3.slider("Hours per week", 1, 40, 8)
    if st.button("Generate roadmap", type="primary"):
        if api.try_call(api.post, "/roadmap/generate", {"target_role": role, "timeline_months": months, "hours_per_week": hours},
                        spinner="Planning..."):
            st.rerun()
api.disclaimer()
