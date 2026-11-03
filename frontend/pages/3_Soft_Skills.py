"""Soft-skills practice: generated HR/behavioural questions, text or voice answers, dual evaluation, follow-ups."""
import api
import streamlit as st
from components import soft_answer_block

user = api.require_login()
st.title("🗣️ Soft-Skills Practice")
st.caption("Questions are generated from your role and resume. Each answer is scored two ways side by side: an LLM "
           "rubric (generative) and a trained classifier (discriminative), plus delivery analytics. At most 2 follow-ups per question.")

with st.sidebar:
    st.subheader("Settings")
    count = st.slider("Number of questions", 1, 6, 3)
    cats = st.multiselect("Categories", ["behavioral", "situational", "hr", "resume"],
                          default=["behavioral", "situational", "hr", "resume"])
    voice = st.toggle("Enable voice answers", value=True, help="Needs faster-whisper; text answers always work.")

if st.button("🎲 Generate questions", type="primary"):
    res = api.try_call(api.post, "/soft/questions", {"count": count, "categories": cats or ["behavioral"],
                                                      "session_id": None}, spinner="Preparing questions...")
    if res:
        st.session_state["soft"] = {"session_id": res["session_id"], "items": [
            {**q, "result": None, "followup": False} for q in res["questions"]], "llm": res.get("llm")}
        st.rerun()

state = st.session_state.get("soft")
if not state:
    st.info("Click **Generate questions** to start a practice session.")
    st.stop()

items = state["items"]
st.caption(f"Session #{state['session_id']}")
api.llm_note(state.get("llm"))
i = 0
while i < len(items):
    item = items[i]
    with st.container(border=True):
        res = soft_answer_block(item, key=str(item["answer_id"]), enable_voice=voice)
        if res:
            item["result"] = res
            fu = res.get("followup")
            if fu:
                items.insert(i + 1, {"answer_id": fu["answer_id"], "question": fu["question"], "category": f"follow-up {fu['depth']}/{fu['max']} · {fu['reason']}",
                                     "result": None, "followup": True})
            st.rerun()
    i += 1

answered = [it for it in items if it["result"]]
if answered:
    avg = sum(it["result"]["score"]["total"] for it in answered) / len(answered)
    st.success(f"Answered {len(answered)} · average score {avg:.0f}/100. Generate your full feedback report on the "
               f"**Feedback** page (session #{state['session_id']}).")
api.disclaimer()
