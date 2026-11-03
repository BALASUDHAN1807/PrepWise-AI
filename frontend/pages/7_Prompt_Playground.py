"""Prompt Playground: edit prompts, change temperature/model, run, validate JSON, save and compare versions.

Demonstrates Unit 3 (prompt engineering) interactively: the same templates the app uses are editable here.
"""
import json

import api
import pandas as pd
import streamlit as st

user = api.require_login()
st.title("🧪 Prompt Playground")
st.caption("Edit a prompt template, change the model or temperature, run it, and see whether the output is valid JSON "
           "for its schema. Runs are rate-limited. Nothing here affects your scores.")

data = api.try_call(api.get, "/prompts")
if not data:
    st.stop()
names = [p["name"] for p in data["prompts"]]
by_name = {p["name"]: p for p in data["prompts"]}

top1, top2 = st.columns([2, 1])
name = top1.selectbox("Prompt template", names)
meta = by_name[name]
source = top2.selectbox("Version", [f"file v{v}" for v in meta["file_versions"]] +
                        [f"saved: {s['label']} (#{s['id']})" for s in meta["saved_versions"]])
if source.startswith("file"):
    tpl = api.try_call(api.get, f"/prompts/{name}?version={source.split('v')[-1]}")
else:
    tpl = api.try_call(api.get, f"/prompts/saved/{source.split('#')[-1].rstrip(')')}")
if not tpl:
    st.stop()
st.caption(meta["description"])
key = f"{name}|{source}"

c1, c2 = st.columns(2)
with c1:
    system = st.text_area("System prompt", tpl["system"], height=200, key=f"sys_{key}")
    user_t = st.text_area("User template (Jinja2: {{ variable }})", tpl["user"], height=220, key=f"usr_{key}")
with c2:
    variables_raw = st.text_area("Variables (JSON)", json.dumps(tpl.get("sample_variables", {}), indent=2, ensure_ascii=False),
                                 height=200, key=f"var_{key}")
    t1, t2, t3 = st.columns(3)
    temp = t1.slider("Temperature", 0.0, 1.5, float(tpl.get("temperature", 0.2)), 0.05, key=f"t_{key}")
    provider = t2.selectbox("Provider", ["(default)", "gemini", "openai", "anthropic", "ollama", "mock"])
    model = t3.text_input("Model (optional)", "")
    schema = st.selectbox("Validate output against schema", ["(none)"] + data["schemas"],
                          index=(data["schemas"].index(tpl["output_schema"]) + 1) if tpl.get("output_schema") in data["schemas"] else 0,
                          key=f"sch_{key}")
    append = st.checkbox("Append the JSON-schema instruction automatically", value=True)

b1, b2, b3 = st.columns(3)
if b1.button("▶ Run", type="primary"):
    try:
        variables = json.loads(variables_raw or "{}")
    except json.JSONDecodeError as exc:
        st.error(f"Variables are not valid JSON: {exc}")
        variables = None
    if variables is not None:
        body = {"system": system, "user_template": user_t, "variables": variables, "temperature": temp,
                "provider": None if provider.startswith("(") else provider, "model": model or None,
                "schema_name": None if schema.startswith("(") else schema, "prompt_name": name, "append_schema": append}
        res = api.try_call(api.post, "/prompts/run", body, spinner="Calling the model...")
        if res:
            st.session_state.setdefault("pg_runs", []).append({"label": f"{source} · T={temp} · {res['provider']}", **res})
runs = st.session_state.get("pg_runs", [])
if runs:
    last = runs[-1]
    st.markdown(f"**Last run** - {last['provider']}/{last['model']} · {last['latency_ms']} ms · T={last['temperature']}")
    ok = last["json_valid"] and last.get("schema_valid") in (True, None)
    (st.success if ok else st.error)(
        f"JSON valid: {last['json_valid']} · schema valid: {last.get('schema_valid')}" +
        (f" · errors: {'; '.join(last['errors'])}" if last["errors"] else ""))
    st.code(last["output"], language="json")
    with st.expander("Rendered user prompt"):
        st.code(last["rendered_user_prompt"], language="text")

with b2.popover("💾 Save as version"):
    label = st.text_input("Version label", "my-variant")
    notes = st.text_input("Notes (optional)")
    if st.button("Save"):
        if api.try_call(api.post, "/prompts/versions", {"name": name, "version_label": label, "system": system,
                                                        "user_template": user_t, "temperature": temp, "notes": notes}):
            st.success("Saved - reload the page to see it in the version list.")
if b3.button("Clear run history"):
    st.session_state["pg_runs"] = []
    st.rerun()

if len(runs) >= 2:
    st.markdown("**Compare runs**")
    df = pd.DataFrame([{"run": i + 1, "setup": r["label"], "latency_ms": r["latency_ms"], "json_valid": r["json_valid"],
                        "schema_valid": r.get("schema_valid"), "output_chars": len(r["output"])} for i, r in enumerate(runs)])
    st.dataframe(df, hide_index=True, width="stretch")
    a, b = st.columns(2)
    ia = a.selectbox("Run A", range(1, len(runs) + 1), index=len(runs) - 2)
    ib = b.selectbox("Run B", range(1, len(runs) + 1), index=len(runs) - 1)
    a.code(runs[ia - 1]["output"], language="json")
    b.code(runs[ib - 1]["output"], language="json")
