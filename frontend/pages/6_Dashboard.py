"""Progress dashboard: Interview Readiness Score, score trend, skill radar, before/after comparison."""
import api
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

user = api.require_login(consent=False)
st.title("📊 Progress Dashboard")
d = api.try_call(api.get, "/dashboard")
if not d:
    st.stop()

r = d["readiness"]
c1, c2 = st.columns([1, 2])
with c1:
    score = r["score"]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score if score is not None else 0,
                                 title={"text": "Interview Readiness"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#16a34a"},
                                        "steps": [{"range": [0, 40], "color": "#fee2e2"}, {"range": [40, 70], "color": "#fef9c3"},
                                                  {"range": [70, 100], "color": "#dcfce7"}]}))
    fig.update_layout(height=280, margin=dict(t=60, b=0))
    st.plotly_chart(fig, width="stretch")
    if score is None:
        st.info("Complete a resume upload, a DSA question or a soft-skills answer to get your score.")
with c2:
    st.markdown("**How this score is built**")
    for k, label in (("resume", "Resume (ATS)"), ("dsa", "DSA"), ("soft", "Soft skills")):
        v = r["components"].get(k)
        w = r["weights"].get(k)
        st.progress((v or 0) / 100, text=f"{label}: " + (f"{v:.0f}/100 (weight {w:.0%})" if v is not None else "no data yet (excluded)"))
    st.caption(r.get("note", ""))

counts = d["counts"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Sessions", counts["sessions"])
m2.metric("DSA attempts", counts["dsa_answers"])
m3.metric("Soft-skill answers", counts["soft_answers"])
m4.metric("Resume ATS", f"{d['resume_ats']:.0f}" if d["resume_ats"] is not None else "-")

left, right = st.columns(2)
with left:
    st.markdown("**Score trend**")
    if d["trend"]:
        df = pd.DataFrame(d["trend"])
        fig = px.line(df, x="date", y="score", color="kind", markers=True, range_y=[0, 100])
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No completed sessions yet.")
with right:
    st.markdown("**Skill radar**")
    if len(d["radar"]) >= 3:
        labels, vals = list(d["radar"]), list(d["radar"].values())
        fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself"))
        fig.update_layout(polar={"radialaxis": {"range": [0, 100]}}, height=320, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("The radar appears after you answer a few questions.")

st.markdown("**Before vs after** (average of your first vs latest sessions)")
ba = d["before_after"]
rows = [{"area": k, "before": v["before"], "after": v["after"]} for k, v in ba.items() if v]
if rows:
    df = pd.DataFrame(rows).melt("area", var_name="when", value_name="score")
    fig = px.bar(df, x="area", y="score", color="when", barmode="group", range_y=[0, 100])
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("Needs at least two sessions in the same area.")
st.caption(d["disclaimer"])
