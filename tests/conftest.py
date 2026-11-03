"""Shared fixtures: isolated SQLite DB, mock LLM provider, in-memory vector store, hash embeddings."""
import os
import tempfile

# Configure BEFORE importing the app so settings pick these up.
_TMP = tempfile.mkdtemp(prefix="prepwise_test_")
os.environ.update({
    "DATABASE_URL": f"sqlite:///{_TMP}/test.db",
    "LLM_PROVIDER": "mock",
    "EMBEDDING_BACKEND": "hash",
    "VECTOR_STORE": "memory",
    "MODELS_DIR": f"{_TMP}/models",
    "ENABLE_TRANSFORMER_SENTIMENT": "false",
    "JWT_SECRET": "test-secret-test-secret-test-secret-0123456789",
    "ADMIN_EMAILS": "admin@example.com",
    "ADMIN_MIN_GROUP": "1",
    "SANDBOX_TIMEOUT_SEC": "5",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.core import ratelimit  # noqa: E402
from backend.core.config import get_settings  # noqa: E402
from backend.core.database import Base, get_engine, reset_engine  # noqa: E402


@pytest.fixture()
def client():
    """Fresh database for every test."""
    get_settings.cache_clear()
    reset_engine()
    from backend.models import db_models  # noqa: F401

    Base.metadata.drop_all(bind=get_engine())
    Base.metadata.create_all(bind=get_engine())
    ratelimit.reset()
    from backend.main import app

    with TestClient(app) as c:
        yield c


def signup(client, email="user@example.com", consent=True, name="Test User"):
    r = client.post("/auth/signup", json={"email": email, "name": name, "password": "password123", "consent": consent})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture()
def auth(client):
    return signup(client)
