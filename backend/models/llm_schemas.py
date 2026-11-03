"""Pydantic models that every LLM JSON response is validated against.

The LLM wrapper retries (max 2) when a response fails validation and then uses
a safe, deterministic fallback, so app logic never consumes unvalidated JSON.
``SCHEMA_REGISTRY`` lets prompt files and the Prompt Playground refer to a
schema by name.
"""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

Score10 = Field(ge=0, le=10)


# ---------------------------------------------------------------- resume ----
class ResumeProject(BaseModel):
    name: str
    description: str = ""
    tech: list[str] = []


class ResumeEducation(BaseModel):
    institution: str = ""
    degree: str = ""
    year: str = ""
    score: str = ""


class ResumeExperience(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    highlights: list[str] = []


class ResumeExtraction(BaseModel):
    name: str = ""
    skills: list[str] = []
    projects: list[ResumeProject] = []
    education: list[ResumeEducation] = []
    experience: list[ResumeExperience] = []
    certifications: list[str] = []
    achievements: list[str] = []


class ResumeSuggestions(BaseModel):
    summary: str
    suggestions: list[str] = Field(min_length=1)


class JDMatchReasoning(BaseModel):
    jd_skills: list[str] = []
    summary: str
    strengths: list[str] = []
    gaps: list[str] = []
    recommendations: list[str] = []


class BulletRewriteItem(BaseModel):
    original: str
    rewritten: str
    rationale: str = ""


class BulletRewrite(BaseModel):
    items: list[BulletRewriteItem] = Field(min_length=1)


class CoverLetter(BaseModel):
    subject: str
    body: str = Field(min_length=50)


# ------------------------------------------------------------------- DSA ----
class Example(BaseModel):
    input: str
    output: str
    explanation: str = ""


class TestCase(BaseModel):
    input: str
    output: str
    hidden: bool = True
    note: str = ""


class GeneratedQuestion(BaseModel):
    """A DSA/SQL question. ``reference_solution`` is executed before serving."""

    title: str
    kind: Literal["code", "sql"] = "code"
    topic: str = ""
    difficulty: str = ""
    statement: str = Field(min_length=20)
    constraints: list[str] = []
    input_format: str
    output_format: str
    examples: list[Example] = Field(min_length=1)
    test_cases: list[TestCase] = Field(min_length=6)
    expected_time_complexity: str
    expected_space_complexity: str
    hints: list[str] = Field(min_length=3, max_length=3)
    reference_solution: str = Field(min_length=10)

    @field_validator("test_cases")
    @classmethod
    def _ensure_visible(cls, v: list[TestCase]) -> list[TestCase]:
        """Guarantee at least two visible and at least two hidden test cases."""
        if not any(not t.hidden for t in v):
            for t in v[:2]:
                t.hidden = False
        if sum(t.hidden for t in v) < 2:
            for t in v[-2:]:
                t.hidden = True
        return v


class CodeReview(BaseModel):
    approach_quality: int = Score10
    approach_comment: str
    time_complexity: str
    space_complexity: str
    complexity_score: int = Score10
    complexity_comment: str = ""
    missed_edge_cases: list[str] = []
    code_quality: int = Score10
    code_quality_comment: str = ""
    better_approach: str = ""


# ----------------------------------------------------------- soft skills ----
class SoftQuestion(BaseModel):
    question: str = Field(min_length=10)
    category: Literal["behavioral", "situational", "hr", "resume"]
    what_good_looks_like: str = ""


class SoftQuestionSet(BaseModel):
    questions: list[SoftQuestion] = Field(min_length=1)


class FollowUp(BaseModel):
    needs_followup: bool
    question: str = ""
    reason: str = ""


class RubricItem(BaseModel):
    score: int = Score10
    reason: str


class StarDetection(BaseModel):
    situation: bool = False
    task: bool = False
    action: bool = False
    result: bool = False
    comment: str = ""


class RubricEvaluation(BaseModel):
    clarity: RubricItem
    structure: RubricItem
    relevance: RubricItem
    confidence: RubricItem
    grammar_vocabulary: RubricItem
    depth: RubricItem
    star: StarDetection = StarDetection()
    overall_comment: str = ""

    RUBRIC_KEYS: ClassVar[tuple[str, ...]] = (
        "clarity",
        "structure",
        "relevance",
        "confidence",
        "grammar_vocabulary",
        "depth",
    )

    def total(self) -> int:
        return sum(getattr(self, k).score for k in self.RUBRIC_KEYS)


class CoTRubricEvaluation(BaseModel):
    """Chain-of-thought variant used in scripts/prompt_comparison.py."""

    reasoning: str
    evaluation: RubricEvaluation


# --------------------------------------------------------------- feedback ----
class PerformanceSummary(BaseModel):
    summary: str
    strengths: list[str] = []
    weaknesses: list[str] = []


class SkillGap(BaseModel):
    skill_id: str
    reason: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class SkillGapMapping(BaseModel):
    skill_gaps: list[SkillGap] = []


class QuestionFeedbackItem(BaseModel):
    question: str
    feedback: str
    model_answer: str = ""


class QuestionFeedback(BaseModel):
    items: list[QuestionFeedbackItem] = []


class ImprovementPlan(BaseModel):
    suggestions: list[str] = Field(min_length=1)


class TranslationBatch(BaseModel):
    items: list[str]


# ---------------------------------------------------------------- roadmap ----
class RoadmapPhase(BaseModel):
    name: str
    weeks: str
    goals: list[str] = []
    skills: list[str] = []


class RoadmapWeek(BaseModel):
    week: int = Field(ge=1)
    focus: str
    tasks: list[str] = Field(min_length=1)


class RoadmapMilestone(BaseModel):
    week: int = Field(ge=1)
    title: str


class RoadmapProject(BaseModel):
    title: str
    description: str = ""
    skills: list[str] = []


class RoadmapCourse(BaseModel):
    course_id: str
    skill: str = ""
    why: str = ""


class RoadmapOutput(BaseModel):
    title: str
    summary: str
    phases: list[RoadmapPhase] = Field(min_length=1)
    weekly_plan: list[RoadmapWeek] = Field(min_length=1)
    milestones: list[RoadmapMilestone] = []
    skills_to_learn: list[str] = []
    projects: list[RoadmapProject] = []
    recommended_courses: list[RoadmapCourse] = []


class GenericJSON(BaseModel):
    """Permissive schema for free-form Playground prompts."""

    model_config = {"extra": "allow"}


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    cls.__name__: cls
    for cls in [
        ResumeExtraction,
        ResumeSuggestions,
        JDMatchReasoning,
        BulletRewrite,
        CoverLetter,
        GeneratedQuestion,
        CodeReview,
        SoftQuestionSet,
        FollowUp,
        RubricEvaluation,
        CoTRubricEvaluation,
        PerformanceSummary,
        SkillGapMapping,
        QuestionFeedback,
        ImprovementPlan,
        TranslationBatch,
        RoadmapOutput,
        GenericJSON,
    ]
}
