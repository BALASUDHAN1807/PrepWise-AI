"""DSA Interview: RAG-grounded questions, code editor, sandbox runs, hints with penalty, LLM review, adaptive difficulty."""
import api
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_ace import st_ace

user = api.require_login()
st.title("💻 DSA Interview")
st.caption("Questions are generated from a local bank by RAG and verified by running the reference solution. "
           "Correctness = test cases executed in a sandbox; the LLM only reviews approach, complexity and style.")

TEMPLATES = {
    "python": "import sys\n\n\ndef main():\n    data = sys.stdin.read().split()\n    # TODO: solve\n    print(0)\n\n\nmain()\n",
    "java": "import java.util.*;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        // TODO: solve\n        System.out.println(0);\n    }\n}\n",
    "cpp": "#include <bits/stdc++.h>\nusing namespace std;\n\nint main() {\n    // TODO: solve\n    cout << 0 << endl;\n    return 0;\n}\n",
    "sql": "-- write one SELECT statement\nSELECT 1;\n",
}
ACE = {"python": "python", "java": "java", "cpp": "c_cpp", "sql": "sql"}

with st.sidebar:
    st.subheader("Question settings")
    topic = st.selectbox("Topic", ["arrays", "strings", "linked_lists", "trees", "graphs", "dp", "sql"])
    diff = st.selectbox("Difficulty", ["adaptive", "easy", "medium", "hard"],
                        help="Adaptive: >=80 on the last question moves up, <40 moves down.")
    style = st.selectbox("Company style", ["(profile default)", "service", "product", "startup"])
    langs = api.try_call(api.get, "/dsa/languages")
    available = [k for k, v in (langs or {}).get("available", {}).items() if v and k != "sql"] or ["python"]
    if langs:
        st.caption(f"Sandbox isolation: {langs['isolation']}")
    sid = st.session_state.get("dsa_session_id")
    if sid and st.button("Start a new practice session"):
        st.session_state.pop("dsa_session_id", None)
        st.session_state.pop("dsa", None)
        st.rerun()

if st.button("🎲 New question", type="primary"):
    body = {"topic": topic, "session_id": st.session_state.get("dsa_session_id"),
            "difficulty": None if diff == "adaptive" else diff,
            "company_style": None if style.startswith("(") else style}
    res = api.try_call(api.post, "/dsa/question", body, spinner="Generating and verifying a question...")
    if res:
        st.session_state["dsa_session_id"] = res["session_id"]
        st.session_state["dsa"] = {"q": res, "hints": [], "result": None, "run": None, "submitted": False}
        st.rerun()

state = st.session_state.get("dsa")
if not state:
    st.info("Pick a topic and click **New question**. Tip: upload your resume first so questions can reflect your skills.")
    st.stop()

q, aid = state["q"]["question"], state["q"]["answer_id"]
left, right = st.columns([5, 6])
with left:
    st.subheader(q["title"])
    st.caption(f"{q['topic']} · **{state['q']['difficulty']}** ({state['q']['difficulty_reason']}) · "
               f"expected {q['expected_time_complexity']} time, {q['expected_space_complexity']} space")
    st.markdown(q["statement"])
    st.markdown("**Input:** " + q["input_format"] + "  \n**Output:** " + q["output_format"])
    if q.get("constraints"):
        st.markdown("**Constraints:** " + "; ".join(q["constraints"]))
    for i, ex in enumerate(q.get("examples", []), 1):
        st.markdown(f"**Example {i}**")
        c1, c2 = st.columns(2)
        c1.code(ex["input"], language="text")
        c2.code(ex["output"], language="text")
        if ex.get("explanation"):
            st.caption(ex["explanation"])
    st.caption(f"{len(q['visible_tests'])} visible tests · {q['hidden_test_count']} hidden tests")
    with st.expander("How was this question generated? (RAG + hallucination guard)"):
        g = state["q"]["generation"]
        st.write(f"Source: **{g['source']}** after {g['attempts']} attempt(s). Retrieved similar bank items: "
                 f"{', '.join(g.get('rag_examples', [])) or '-'}")
        guard = g.get("guard") or {}
        st.write(f"Guard: reference solution passed {guard.get('passed')}/{guard.get('total')} of its own tests"
                 + ("" if guard.get("ok") else f" - rejected ({guard.get('reason')})"))
        for r in g.get("rejections", []):
            st.caption(f"Rejected attempt: {r}")
        api.llm_note(g.get("llm"))

    st.markdown("---")
    hints_total = q["hint_count"]
    if st.button(f"💡 Get hint ({len(state['hints'])}/{hints_total} used)", disabled=len(state["hints"]) >= hints_total
                 or state["submitted"]):
        h = api.try_call(api.post, f"/dsa/{aid}/hint")
        if h:
            state["hints"].append(h)
            st.rerun()
    for h in state["hints"]:
        st.info(f"Hint {h['hint_number']}: {h['hint']}")
    if hints_total:
        st.caption("Each hint costs points off your final score (shown in the breakdown).")

with right:
    is_sql = q.get("kind") == "sql"
    lang = "sql" if is_sql else st.selectbox("Language", available, key=f"lang_{aid}")
    code = st_ace(value=TEMPLATES[lang], language=ACE[lang], theme="monokai", height=380, font_size=14,
                  key=f"ace_{aid}_{lang}", auto_update=True, wrap=True)
    stdin = st.text_area("Custom input (optional - runs your code once with this stdin)", height=70, key=f"stdin_{aid}") \
        if not is_sql else None
    b1, b2 = st.columns(2)
    run = b1.button("▶ Run visible tests", disabled=state["submitted"])
    sub = b2.button("✅ Submit", type="primary", disabled=state["submitted"])
    body = {"language": lang, "code": code or TEMPLATES[lang], "stdin": stdin or None}
    if run:
        state["run"] = api.try_call(api.post, f"/dsa/{aid}/run", body, spinner="Running in sandbox...")
    if sub:
        res = api.try_call(api.post, f"/dsa/{aid}/submit", body, spinner="Running all tests and reviewing...")
        if res:
            state["result"], state["submitted"] = res, True

    def show_cases(summary: dict, label: str) -> None:
        st.markdown(f"**{label}: {summary['passed']}/{summary['total']} passed**")
        rows = [{"#": c["index"] + 1, "type": "hidden" if c["hidden"] else "visible", "status": c["status"],
                 "ms": c["runtime_ms"], "input": c["input"][:60] if not c["hidden"] else "(hidden)",
                 "expected": c["expected"][:40] if not c["hidden"] else "", "actual": c["actual"][:40] if not c["hidden"] else ""}
                for c in summary["cases"]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        for c in summary["cases"]:
            if c.get("error"):
                st.code(c["error"][-800:], language="text")
                break

    if state["run"] and not state["submitted"]:
        show_cases(state["run"]["summary"], "Visible tests")
        if state["run"].get("custom"):
            cu = state["run"]["custom"]
            st.markdown("**Custom run output**")
            st.code(cu["stdout"] or "(no output)", language="text")
            if cu["stderr"]:
                st.code(cu["stderr"], language="text")
            if cu["timed_out"]:
                st.warning("Time limit exceeded.")

res = state["result"]
if res:
    st.markdown("---")
    sc = res["score"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Final score", f"{sc['total']:.0f} / 100")
    s2.metric("Tests passed", f"{res['tests']['passed']}/{res['tests']['total']}")
    s3.metric("Next difficulty", res["next_difficulty"])
    comp = pd.DataFrame({"component": list(sc["components"]), "points": list(sc["components"].values()),
                         "max": [sc["weights"][k] * 100 for k in sc["components"]]})
    fig = px.bar(comp, x="points", y="component", orientation="h", text="points", range_x=[0, 60])
    fig.update_layout(height=230, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Formula: 60% tests + 20% logic + 10% complexity + 10% quality − {sc['hint_penalty']:.0f} hint penalty "
               f"({sc['hints_used']} hint(s)). LLM review scores are capped by the test pass-rate "
               f"(logic after cap: {sc['logic_after_cap']}/10).")
    show_cases(res["tests"], "All tests")
    rv = res["review"]
    with st.expander("🧠 AI code review (logic, complexity, style)", expanded=True):
        st.write(f"**Approach {rv['approach_quality']}/10** - {rv['approach_comment']}")
        st.write(f"**Complexity** {rv['time_complexity']} time / {rv['space_complexity']} space "
                 f"({rv['complexity_score']}/10) - {rv['complexity_comment']}")
        st.write(f"**Code quality {rv['code_quality']}/10** - {rv['code_quality_comment']}")
        if rv.get("missed_edge_cases"):
            st.markdown("**Possible missed edge cases:** " + "; ".join(rv["missed_edge_cases"]))
        if rv.get("better_approach"):
            st.markdown(f"**Better approach:** {rv['better_approach']}")
        api.llm_note(res.get("review_llm"))
    if res.get("reference_solution"):
        with st.expander("Reference solution"):
            st.code(res["reference_solution"], language="python")
    st.caption(res.get("disclaimer", ""))
else:
    api.disclaimer()
