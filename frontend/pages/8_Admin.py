"""Admin dashboard: anonymised, aggregated batch insights (placement-cell view). Admin accounts only."""
import api
import pandas as pd
import plotly.express as px
import streamlit as st

user = api.require_login(consent=False)
st.title("🛡️ Admin - Batch Overview")
if user.get("role") != "admin":
    st.error("This page is for admin accounts. Admins are configured with ADMIN_EMAILS in the server's .env file.")
    st.stop()

d = api.try_call(api.get, "/admin/overview")
if not d:
    st.stop()
b = d["batch"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Consenting students", b["consenting_users"])
c2.metric("With activity", b["with_activity"])
c3.metric("Avg readiness", b["avg_readiness"] if b["avg_readiness"] is not None else "-")
c4.metric("Avg ATS", b["avg_ats"] if b["avg_ats"] is not None else "-")
pv = d["privacy"]
st.info(f"🔒 {pv['note']} Anonymised IDs are salted hashes; groups smaller than {pv['min_group']} are suppressed.")

bins = d["distributions"]["bins"]
frames = []
for key, label in (("readiness", "Readiness"), ("dsa", "DSA"), ("soft", "Soft skills")):
    for bn in bins:
        frames.append({"metric": label, "score band": bn, "students": d["distributions"][key].get(bn, 0)})
fig = px.bar(pd.DataFrame(frames), x="score band", y="students", color="metric", barmode="group")
fig.update_layout(height=340, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.markdown("**Most common skill gaps**")
    if d["common_skill_gaps"]:
        df = pd.DataFrame(d["common_skill_gaps"])
        fig = px.bar(df, x="share_pct", y="name", orientation="h", text="students")
        fig.update_layout(height=320, yaxis={"autorange": "reversed"}, xaxis_title="% of active students")
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No feedback reports yet.")
with right:
    st.markdown("**Top improvers (anonymised)**")
    if d["top_improvers"]:
        st.dataframe(pd.DataFrame(d["top_improvers"]), hide_index=True, width="stretch")
    elif pv["per_student_views_suppressed"]:
        st.caption("Hidden: the batch is below the minimum group size.")
    else:
        st.caption("Needs students with at least two sessions.")

if st.button("Prepare CSV export"):
    try:
        csv = api.call("GET", "/admin/export.csv", raw=True)
        st.download_button("⬇️ Download anonymised CSV", csv, "prepwise-batch.csv", "text/csv")
    except api.ApiError as exc:
        st.warning(str(exc))
