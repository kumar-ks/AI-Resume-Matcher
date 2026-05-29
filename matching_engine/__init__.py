"""
AI Matching Engine - Resume to Job Description Matching Pipeline

This module implements the 6-stage AI matching pipeline:
1. JD Understanding - Extract requirements from job descriptions
2. Resume Understanding - Parse and extract candidate information
3. Semantic Matching - Compare resume against JD using embeddings
4. Scoring - Weighted scoring model for qualification percentage
5. Explainability - Generate human-readable reasoning
6. Output - Final structured match results

Dependencies:
    - litellm (LLM provider interface)
    - sentence-transformers (semantic embeddings)
    - pydantic (data models)
"""

from matching_engine.models import (
    JobDescription,
    ResumeProfile,
    MatchResult,
    ScoringBreakdown,
    ExplainabilityReport,
)
from matching_engine.pipeline import MatchingPipeline

__all__ = [
    "JobDescription",
    "ResumeProfile",
    "MatchResult",
    "ScoringBreakdown",
    "ExplainabilityReport",
    "MatchingPipeline",
]
