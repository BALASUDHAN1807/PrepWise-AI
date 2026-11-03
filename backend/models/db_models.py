"""ORM models: users, resumes, sessions, answers, feedback, roadmaps, progress, prompt_versions.

All child tables use ``ON DELETE CASCADE`` so the "delete all my data" action
removes every row that belongs to a user.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="user")  # user | admin
    target_role: Mapped[str] = mapped_column(String(120), default="Software Engineer")
    company_type: Mapped[str] = mapped_column(String(20), default="product")  # service|product|startup
    experience_level: Mapped[str] = mapped_column(String(20), default="fresher")
    preferred_language: Mapped[str] = mapped_column(String(10), default="en")  # en|ta|hi
    consent_given: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    resumes = relationship("Resume", cascade="all, delete-orphan", passive_deletes=True)
    sessions = relationship("InterviewSession", cascade="all, delete-orphan", passive_deletes=True)
    roadmaps = relationship("Roadmap", cascade="all, delete-orphan", passive_deletes=True)


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    raw_text: Mapped[str] = mapped_column(Text)  # never logged; deleted with the user
    parsed_json: Mapped[dict] = mapped_column(JSON, default=dict)
    ats_score: Mapped[float] = mapped_column(Float, default=0.0)
    ats_breakdown: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InterviewSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # dsa | softskills | mock
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | completed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_limit_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    answers = relationship("Answer", cascade="all, delete-orphan", passive_deletes=True)
    feedback = relationship("Feedback", cascade="all, delete-orphan", passive_deletes=True)


class Answer(Base):
    """One question + the user's answer + every evaluation artifact."""

    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    question_type: Mapped[str] = mapped_column(String(10))  # dsa | soft
    topic: Mapped[str] = mapped_column(String(60), default="")
    difficulty: Mapped[str] = mapped_column(String(10), default="")
    question_json: Mapped[dict] = mapped_column(JSON, default=dict)  # includes hidden tests
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    code: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(20), nullable=True)
    test_results: Mapped[dict] = mapped_column(JSON, default=dict)
    evaluation: Mapped[dict] = mapped_column(JSON, default=dict)  # LLM review / rubric
    classifier: Mapped[dict] = mapped_column(JSON, default=dict)
    analytics: Mapped[dict] = mapped_column(JSON, default=dict)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_breakdown: Mapped[dict] = mapped_column(JSON, default=dict)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    followup_of: Mapped[int | None] = mapped_column(Integer, nullable=True)
    followup_depth: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    language: Mapped[str] = mapped_column(String(10), default="en")
    content: Mapped[dict] = mapped_column(JSON, default=dict)  # in requested language
    content_en: Mapped[dict] = mapped_column(JSON, default=dict)  # canonical English version
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Roadmap(Base):
    __tablename__ = "roadmaps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    target_role: Mapped[str] = mapped_column(String(120))
    timeline_months: Mapped[int] = mapped_column(Integer)
    hours_per_week: Mapped[int] = mapped_column(Integer)
    skill_gaps: Mapped[list] = mapped_column(JSON, default=list)
    content: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    progress = relationship("RoadmapProgress", cascade="all, delete-orphan", passive_deletes=True)


class RoadmapProgress(Base):
    __tablename__ = "progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    roadmap_id: Mapped[int] = mapped_column(ForeignKey("roadmaps.id", ondelete="CASCADE"), index=True)
    item_key: Mapped[str] = mapped_column(String(120))
    done: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PromptVersion(Base):
    """Prompt versions saved from the Prompt Playground."""

    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    version_label: Mapped[str] = mapped_column(String(40))
    system: Mapped[str] = mapped_column(Text)
    user_template: Mapped[str] = mapped_column(Text)
    temperature: Mapped[float] = mapped_column(Float, default=0.2)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
