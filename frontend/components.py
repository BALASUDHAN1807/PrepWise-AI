"""Reusable Streamlit widgets shared by the soft-skills and mock-interview pages."""
from __future__ import annotations

import api
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RUBRIC = ["clarity", "structure", "relevance", "confidence", "grammar_vocabulary", "depth"]


def soft_result(res: dict) -> None:
    """Show the rubric (LLM) and the classifier (discriminative) side by side, plus delivery analytics."""
    sc = res["score"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall", f"{sc['total']:.0f}/100")
    c2.metric("LLM rubric", f"{sc['components']['rubric']:.0f}/100")
    cls = res.get("classifier")
    c3.metric("Classifier", cls["label"] if cls else "n/a",
              f"{cls['confidence']:.0%} confident" if cls else None, delta_color="off")
    c4.metric("Delivery", f"{sc['components']['delivery']:.0f}/100")

    left, right = st.columns(2)
    rub = res["rubric"]
    with left:
        st.markdown("**Generative: LLM rubric (0-10, with reasons)**")
        vals = [rub[k]["score"] for k in RUBRIC]
        fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=[k.replace("_", " ").title() for k in RUBRIC] +
                                        [RUBRIC[0].title()], fill="toself"))
        fig.update_layout(polar={"radialaxis": {"range": [0, 10]}}, height=300, margin=dict(t=20, b=20, l=40, r=40),
                          showlegend=False)
        st.plotly_chart(fig, width="stretch")
        for k in RUBRIC:
            st.caption(f"**{k.replace('_', ' ').title()} {rub[k]['score']}/10** - {rub[k]['reason']}")
        star = rub.get("star", {})
        st.markdown("STAR: " + " ".join(f"{'✅' if star.get(k) else '❌'} {k.title()}"
                                        for k in ("situation", "task", "action", "result")))
        if rub.get("overall_comment"):
            st.info(rub["overall_comment"])
        api.llm_note(res.get("rubric_llm"))
    with right:
        st.markdown("**Discriminative: trained classifier (poor / average / good)**")
        if cls:
            probs = cls["probabilities"]
            order = [k for k in ("poor", "average", "good") if k in probs]
            fig = go.Figure(go.Bar(x=order, y=[probs[k] for k in order], text=[f"{probs[k]:.0%}" for k in order]))
            fig.update_layout(height=220, yaxis={"range": [0, 1]}, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Model: {cls['model']}" + (f" · DistilBERT: {cls['distilbert']}" if cls.get("distilbert") else ""))
            st.caption("The classifier is trained on a small synthetic dataset; treat it as a second opinion, not ground truth.")
        else:
            st.info("Classifier model not trained yet - run `python scripts/train_classifier.py`.")
        an = res["analytics"]
        st.markdown("**Delivery analytics**")
        rows = {"Words": an["word_count"], "Sentences": an["sentence_count"],
                "Words per minute": an["wpm"] if an["wpm"] is not None else "(voice only)",
                "Pace": an["pace"] or "(voice only)",
                "Filler words": f"{an['filler_count']} ({an['filler_rate_per_100_words']}/100 words)",
                "Sentiment": f"{an['sentiment']['label']} ({an['sentiment']['method']})"}
        if an.get("pauses"):
            rows["Long pauses"] = f"{an['pauses']['count']} (longest {an['pauses']['longest_sec']}s)"
        st.dataframe(pd.DataFrame({"metric": list(rows), "value": [str(v) for v in rows.values()]}),
                     hide_index=True, width="stretch")
        if an["fillers"]:
            st.caption("Fillers: " + ", ".join(f"{w}×{n}" for w, n in an["fillers"].items()))
    with st.expander("How the overall score was computed"):
        w = sc["weights"]
        st.write(f"{w['rubric']:.0%} rubric + {w['classifier']:.0%} classifier expectation + {w['delivery']:.0%} delivery")
        st.json(sc)
    st.caption(res.get("disclaimer", api.DISCLAIMER))


def soft_answer_block(item: dict, key: str, enable_voice: bool = True) -> dict | None:
    """Render one question with text/voice input. Returns the API result when the answer was just submitted."""
    st.markdown(f"##### {'↳ Follow-up: ' if item.get('followup') else ''}{item['question']}")
    if item.get("category"):
        st.caption(item["category"])
    if item.get("result"):
        soft_result(item["result"])
        return None
    text_key = f"txt_{key}"
    duration = segments = None
    if enable_voice:
        audio = st.audio_input("🎙️ Record your answer (optional - transcribed with Whisper)", key=f"aud_{key}")
        if audio is not None and st.session_state.get(f"aud_done_{key}") != audio.size:
            tr = api.try_call(api.call, "POST", "/soft/transcribe",
                              files={"audio": ("answer.wav", audio.getvalue(), "audio/wav")}, spinner="Transcribing...")
            if tr:
                st.session_state[text_key] = tr["text"]
                st.session_state[f"dur_{key}"] = tr.get("duration_sec")
                st.session_state[f"seg_{key}"] = tr.get("segments")
                st.session_state[f"aud_done_{key}"] = audio.size
                st.rerun()
        duration, segments = st.session_state.get(f"dur_{key}"), st.session_state.get(f"seg_{key}")
        if duration:
            st.caption(f"Transcript below is editable. Duration {duration:.0f}s is used for pace and pause analytics.")
    text = st.text_area("Your answer", key=text_key, height=160,
                        placeholder="Use the STAR method: Situation, Task, Action, Result.")
    if st.button("Submit answer", type="primary", key=f"sub_{key}"):
        return api.try_call(api.post, f"/soft/{item['answer_id']}/answer",
                            {"text": text, "duration_sec": duration, "segments": segments}, spinner="Evaluating...")
    return None
