"""
Data models for the AI Matching Engine.

Defines structured representations for job descriptions, resumes,
scoring breakdowns, and match results used throughout the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ExperienceLevel(str, Enum):
    """Experience level categories."""
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class SkillCategory(str, Enum):
    """Skill requirement categories from JD."""
    MUST_HAVE = "must_have"
    GOOD_TO_HAVE = "good_to_have"


class Skill(BaseModel):
    """A single skill with metadata."""
    name: str
    category: SkillCategory = SkillCategory.MUST_HAVE
    years_required: Optional[float] = None
    proficiency_level: Optional[str] = None


class JobDescription(BaseModel):
    """
    Structured representation of a parsed Job Description.

    Stage 1 output: JD Understanding extracts these fields from
    raw JD text (PDF/DOCX/plain text).
    """
    title: str = ""
    must_have_skills: list[Skill] = Field(default_factory=list)
    good_to_have_skills: list[Skill] = Field(default_factory=list)
    experience_range_min: Optional[float] = None  # years
    experience_range_max: Optional[float] = None  # years
    education: list[str] = Field(default_factory=list)
    domain_industry: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    role_level: Optional[ExperienceLevel] = None
    raw_text: str = ""


class Project(BaseModel):
    """A project from a candidate's resume."""
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    duration_months: Optional[int] = None
    role: Optional[str] = None


class WorkExperience(BaseModel):
    """A single work experience entry."""
    company: str = ""
    title: str = ""
    duration_months: Optional[int] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    is_current: bool = False
    technologies: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    domain: Optional[str] = None


class ResumeProfile(BaseModel):
    """
    Structured representation of a parsed Resume.

    Stage 2 output: Resume Understanding extracts these fields
    from raw resume text.
    """
    first_name: str = ""
    middle_name: Optional[str] = None
    last_name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    career_summary: str = ""
    skills: list[str] = Field(default_factory=list)
    total_experience_years: Optional[float] = None
    work_experiences: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    domain_expertise: list[str] = Field(default_factory=list)
    raw_text: str = ""

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        return " ".join(p for p in parts if p)


class SemanticMatchResult(BaseModel):
    """
    Stage 3 output: Semantic Matching scores.

    Captures contextual similarity, skill relevance, role alignment,
    domain relevance, technology mapping, and experience alignment.
    """
    contextual_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    skill_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    role_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    technology_mapping: float = Field(default=0.0, ge=0.0, le=1.0)
    experience_alignment: float = Field(default=0.0, ge=0.0, le=1.0)


class ScoringBreakdown(BaseModel):
    """
    Stage 4 output: Weighted scoring model breakdown.

    Each component contributes to the final qualification percentage.
    """
    must_have_match: float = Field(default=0.0, ge=0.0, le=1.0)
    experience_match: float = Field(default=0.0, ge=0.0, le=1.0)
    skills_depth: float = Field(default=0.0, ge=0.0, le=1.0)
    project_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    recency_factor: float = Field(default=0.0, ge=0.0, le=1.0)
    qualification_percentage: float = Field(default=0.0, ge=0.0, le=100.0)

    # Weights for each scoring component
    class Config:
        json_schema_extra = {
            "weights": {
                "must_have_match": 0.35,
                "experience_match": 0.25,
                "skills_depth": 0.20,
                "project_relevance": 0.12,
                "recency_factor": 0.08,
            }
        }


class ExplainabilityReport(BaseModel):
    """
    Stage 5 output: Human-readable explanation of the match.

    Provides reasoning, matched strengths, missing skills,
    improvement areas, and a recommendation.
    """
    reason_for_score: str = ""
    matched_strengths: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    improvement_areas: list[str] = Field(default_factory=list)
    recommendation: str = ""


class MatchResult(BaseModel):
    """
    Stage 6 output: Final structured output of the matching engine.

    Combines all pipeline stages into a single result per candidate.
    """
    candidate: ResumeProfile
    job_description: JobDescription
    qualification_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    semantic_scores: SemanticMatchResult = Field(default_factory=SemanticMatchResult)
    scoring_breakdown: ScoringBreakdown = Field(default_factory=ScoringBreakdown)
    explainability: ExplainabilityReport = Field(default_factory=ExplainabilityReport)
    key_strengths: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    reasoning: str = ""
    recommendation: str = ""
