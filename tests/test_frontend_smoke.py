"""Streamlit pages render against a real (threaded) API server without exceptions.

Uses streamlit's AppTest runner - no browser needed. The backend runs with the mock LLM (see conftest).
"""
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


@pytest.fixture()
def live_api(client, monkeypatch):
    """Serve the app on a free port in a background thread (the `client` fixture resets the DB first)."""
    from backend.main import app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    monkeypatch.setenv("BACKEND_URL", f"http://127.0.0.1:{port}")
    monkeypatch.syspath_prepend(str(FRONTEND))
    for m in ("api", "components"):  # re-import so they pick up this test's BACKEND_URL
        sys.modules.pop(m, None)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def login(base, email):
    body = {"email": email, "name": "UI Tester", "password": "password123", "consent": True}
    r = httpx.post(f"{base}/auth/signup", json=body)
    return r.json()["token"], r.json()["user"]


PAGES = ["app.py", "pages/1_Resume.py", "pages/2_DSA_Interview.py", "pages/3_Soft_Skills.py", "pages/4_Feedback.py",
         "pages/5_Roadmap.py", "pages/6_Dashboard.py", "pages/7_Prompt_Playground.py"]


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_when_logged_in(live_api, page):
    token, user = login(live_api, f"{page.replace('/', '_').replace('.py', '')}@example.com")
    at = AppTest.from_file(str(FRONTEND / page), default_timeout=60)
    at.session_state["token"], at.session_state["user"] = token, user
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_pages_ask_for_login_when_anonymous(live_api):
    at = AppTest.from_file(str(FRONTEND / "pages/2_DSA_Interview.py"), default_timeout=30).run()
    assert not at.exception and any("log in" in i.value.lower() for i in at.info)


def test_admin_page_blocks_normal_users_and_shows_admin_overview(live_api):
    token, user = login(live_api, "plain@example.com")
    at = AppTest.from_file(str(FRONTEND / "pages/8_Admin.py"), default_timeout=30)
    at.session_state["token"], at.session_state["user"] = token, user
    at.run()
    assert any("admin accounts" in e.value for e in at.error)
    atoken, admin = login(live_api, "admin@example.com")
    at = AppTest.from_file(str(FRONTEND / "pages/8_Admin.py"), default_timeout=30)
    at.session_state["token"], at.session_state["user"] = atoken, admin
    at.run()
    assert not at.exception and any(m.label == "Consenting students" for m in at.metric)


def test_dsa_and_soft_flows_through_the_ui(live_api):
    token, user = login(live_api, "flow@example.com")
    at = AppTest.from_file(str(FRONTEND / "pages/2_DSA_Interview.py"), default_timeout=90)
    at.session_state["token"], at.session_state["user"] = token, user
    at.run()
    next(b for b in at.button if "New question" in b.label).click().run()
    assert not at.exception and at.subheader
    at = AppTest.from_file(str(FRONTEND / "pages/3_Soft_Skills.py"), default_timeout=90)
    at.session_state["token"], at.session_state["user"] = token, user
    at.run()
    next(b for b in at.button if "Generate questions" in b.label).click().run()
    at.text_area[0].set_value("In my project I led four people, split the work, and we shipped on time. I learned to communicate early.").run()
    next(b for b in at.button if b.label == "Submit answer").click().run()
    assert not at.exception and any(m.label == "Overall" for m in at.metric)
