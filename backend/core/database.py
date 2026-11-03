"""SQLAlchemy engine/session setup for the SQLite database."""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


_engine = None
_SessionLocal: sessionmaker | None = None


def _build_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    if url.startswith("sqlite"):
        # Enforce ON DELETE CASCADE so "delete all my data" really removes child rows.
        @event.listens_for(engine, "connect")
        def _fk_pragma(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def get_engine():
    """Lazily create the global engine from settings."""
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _build_engine(get_settings().database_url)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def reset_engine(url: str | None = None) -> None:
    """Rebuild the engine (used by tests to point at a temporary database)."""
    global _engine, _SessionLocal
    _engine = _build_engine(url or get_settings().database_url)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they do not exist."""
    from backend.models import db_models  # noqa: F401  (register models)

    Base.metadata.create_all(bind=get_engine())


def SessionLocal() -> Session:  # noqa: N802 - factory naming kept for familiarity
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a DB session that is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
