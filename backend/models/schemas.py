"""Request/response schemas for the REST API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

Language = Literal["en", "ta", "hi"]


class SignupIn(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=128)
    consent: bool = False


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class ProfileIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    target_role: str | None = Field(default=None, min_length=2, max_length=120)
    company_type: Literal["service", "product", "startup"] | None = None
    experience_level: Literal["fresher", "junior", "mid"] | None = None
    preferred_language: Language | None = None


class ConsentIn(BaseModel):
    consent: bool


class DeleteIn(BaseModel):
    confirm: bool = False


class JDIn(BaseModel):
    jd_text: str = Field(min_length=40, max_length=12000)


class BulletsIn(BaseModel):
    bullets: list[str] = Field(min_length=1, max_length=10)
    target_role: str | None = None


class CoverLetterIn(BaseModel):
    jd_text: str = Field(min_length=40, max_length=12000)
    company: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=120)
    tone: Literal["professional", "enthusiastic", "concise"] = "professional"


class SessionIn(BaseModel):
    kind: Literal["dsa", "softskills", "mock"] = "dsa"
    time_limit_min: int | None = Field(default=None, ge=5, le=180)
    dsa_count: int = Field(default=2, ge=1, le=5)
    hr_count: int = Field(default=3, ge=1, le=8)


class DSAQuestionIn(BaseModel):
    session_id: int | None = None
    topic: Literal["arrays", "strings", "linked_lists", "trees", "graphs", "dp", "sql"] = "arrays"
    difficulty: Literal["easy", "medium", "hard"] | None = None  # None -> adaptive
    company_style: Literal["service", "product", "startup"] | None = None


class CodeIn(BaseModel):
    language: Literal["python", "java", "cpp", "sql"] = "python"
    code: str = Field(min_length=1, max_length=50_000)
    stdin: str | None = Field(default=None, max_length=20_000)


class SoftQuestionsIn(BaseModel):
    session_id: int | None = None
    count: int = Field(default=3, ge=1, le=8)
    categories: list[Literal["behavioral", "situational", "hr", "resume"]] = ["behavioral", "situational", "hr", "resume"]


class SoftAnswerIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    duration_sec: float | None = Field(default=None, gt=0, le=1800)
    segments: list[dict] | None = None

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Answer cannot be blank")
        return v


class TTSIn(BaseModel):
    text: str = Field(min_length=1, max_length=1500)
    lang: Language = "en"


class FeedbackIn(BaseModel):
    language: Language | None = None


class RoadmapIn(BaseModel):
    target_role: str | None = Field(default=None, min_length=2, max_length=120)
    timeline_months: Literal[3, 6, 12] = 6
    hours_per_week: int = Field(default=8, ge=1, le=80)


class ProgressIn(BaseModel):
    item_key: str = Field(min_length=1, max_length=120)
    done: bool


class PromptRunIn(BaseModel):
    system: str = Field(max_length=8000)
    user_template: str = Field(min_length=1, max_length=12000)
    variables: dict = {}
    temperature: float = Field(default=0.2, ge=0, le=1.5)
    provider: Literal["gemini", "openai", "anthropic", "ollama", "mock"] | None = None
    model: str | None = Field(default=None, max_length=80)
    schema_name: str | None = None
    prompt_name: str | None = None
    append_schema: bool = True


class PromptVersionIn(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    version_label: str = Field(min_length=1, max_length=40)
    system: str = Field(max_length=8000)
    user_template: str = Field(min_length=1, max_length=12000)
    temperature: float = Field(default=0.2, ge=0, le=1.5)
    notes: str = Field(default="", max_length=1000)
