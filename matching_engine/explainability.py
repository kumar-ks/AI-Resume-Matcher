"""
Stage 5: Explainability
========================

PURPOSE:
    Generates human-readable reasoning for the match score.
    Provides reason for score, matched strengths, missing skills,
    improvement areas, and a recommendation.

IMPLEMENTATION:
    Uses a LangChain chain (prompt → ChatLiteLLMModel → PydanticOutputParser) for
    structured explanation generation with automatic retry. Falls back to rule-based
    explanation generation if the chain fails.

CALL HIERARCHY:
    Called by:
        - pipeline.py → calls ExplainabilityEngine.explain() as Stage 5
    Calls internally:
        - create_explain_chain() → LangChain chain for explanation
        - _fallback_explanation() → rule-based explanation when chain fails
"""

import logging
from typing import Any, Optional

from matching_engine.chains.explain_chain import ExplainOutput, create_explain_chain
from matching_engine.chains.factory import get_llm_for_stage
from matching_engine.models import (
    ExplainabilityReport,
    JobDescription,
    ResumeProfile,
    ScoringBreakdown,
    SemanticMatchResult,
)
from matching_engine.observability import (
    create_generation,
    end_generation,
)

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Generates human-readable explanations for match scores.

    Uses a LangChain chain with PydanticOutputParser to produce structured
    explanations. Falls back to rule-based explanations when the chain fails.
    """

    def __init__(self, model: str = "ollama/llama2", temperature: float = 0.3):
        """
        Initialize Explainability Engine.

        Args:
            model: LiteLLM model identifier
            temperature: LLM temperature (slightly higher for natural language)
        """
        self.model = model
        self.temperature = temperature
        self._trace_parent: Optional[Any] = None

        # Create the LangChain chain (uses factory for gateway routing)
        explain_llm = get_llm_for_stage("explain", model=model, temperature=temperature)
        self._chain = create_explain_chain(
            model=model,
            temperature=temperature,
            max_tokens=2048,
            timeout=300,
            max_retries=2,
            llm=explain_llm,
        )

        logger.debug(f"ExplainabilityEngine initialized with model={model}, temperature={temperature}")

    async def explain(
        self,
        jd: JobDescription,
        resume: ResumeProfile,
        scoring: ScoringBreakdown,
        semantic: SemanticMatchResult,
        trace_parent: Optional[Any] = None,
    ) -> ExplainabilityReport:
        """
        Generate explanation for the match result.

        Flow:
            1. Build chain input with all match data
            2. Invoke LangChain chain (with retry on parse failure)
            3. Convert chain output → ExplainabilityReport
            4. On failure, generate rule-based fallback explanation

        Args:
            jd: Parsed job description (from Stage 1)
            resume: Parsed resume profile (from Stage 2)
            scoring: Scoring breakdown from Stage 4
            semantic: Semantic match result from Stage 3
            trace_parent: Optional LangFuse span/trace for observability

        Returns:
            ExplainabilityReport with reasoning and recommendations
        """
        logger.info(f"Stage 5: Generating explanation for {resume.full_name}")
        parent = trace_parent or self._trace_parent

        generation = create_generation(
            parent=parent,
            name="explainability-chain",
            model=self.model,
            input_data={
                "candidate_name": resume.full_name,
                "qualification_percentage": scoring.qualification_percentage,
            },
            model_parameters={"temperature": self.temperature},
            metadata={"method": "langchain_chain"},
        )

        try:
            # Build chain input
            chain_input = self._build_chain_input(jd, resume, scoring)

            # Invoke the LangChain chain
            result: ExplainOutput = await self._chain.ainvoke(chain_input)

            logger.debug(f"Chain returned: recommendation='{result.recommendation}'")
            end_generation(generation, output=result.model_dump())

            # Convert to ExplainabilityReport
            return ExplainabilityReport(
                reason_for_score=result.reason_for_score,
                matched_strengths=result.matched_strengths,
                missing_skills=result.missing_skills,
                improvement_areas=result.improvement_areas,
                recommendation=result.recommendation,
            )

        except Exception as e:
            logger.error(f"Explainability chain failed: {e}")
            end_generation(generation, level="ERROR", status_message=str(e))
            return self._fallback_explanation(jd, resume, scoring)

    def _build_chain_input(
        self,
        jd: JobDescription,
        resume: ResumeProfile,
        scoring: ScoringBreakdown,
    ) -> dict:
        """Build the input dict for the explain chain."""
        must_have_skills = ", ".join(s.name for s in jd.must_have_skills) or "Not specified"
        candidate_skills = ", ".join(resume.skills[:15]) or "Not extracted"

        exp_min = jd.experience_range_min or 0
        exp_max = jd.experience_range_max or "Not specified"
        exp_required = f"{exp_min}-{exp_max} years" if exp_max != "Not specified" else "Not specified"

        return {
            "job_title": jd.title or "Not specified",
            "candidate_name": resume.full_name or "Unknown",
            "score": scoring.qualification_percentage,
            "must_have_skills": must_have_skills,
            "candidate_skills": candidate_skills,
            "exp_required": exp_required,
            "candidate_exp": resume.total_experience_years or "Unknown",
            "must_have_score": scoring.must_have_match,
            "exp_score": scoring.experience_match,
            "skills_depth": scoring.skills_depth,
            "project_score": scoring.project_relevance,
            "recency_score": scoring.recency_factor,
        }

    def _fallback_explanation(
        self,
        jd: JobDescription,
        resume: ResumeProfile,
        scoring: ScoringBreakdown,
    ) -> ExplainabilityReport:
        """
        Generate a rule-based explanation when the LangChain chain fails.

        Uses scoring breakdown to produce deterministic explanations.
        """
        logger.info("Using fallback rule-based explanation")

        # Identify matched strengths and missing skills
        candidate_skills_lower = {s.lower() for s in resume.skills}
        matched_strengths = []
        missing_skills = []

        for skill in jd.must_have_skills:
            if skill.name.lower() in candidate_skills_lower:
                matched_strengths.append(skill.name)
            else:
                missing_skills.append(skill.name)

        # Map score to recommendation tier
        score = scoring.qualification_percentage
        if score >= 85:
            recommendation = "Strong Fit - Recommended for next round."
        elif score >= 70:
            recommendation = "Good Fit - Consider for interview."
        elif score >= 50:
            recommendation = "Partial Fit - May need additional screening."
        else:
            recommendation = "Weak Fit - Does not meet key requirements."

        # Build reason string from scoring dimensions
        reason_parts = []
        if scoring.must_have_match >= 0.8:
            reason_parts.append("Strong match in core required skills")
        elif scoring.must_have_match >= 0.5:
            reason_parts.append("Moderate match in required skills")
        else:
            reason_parts.append("Limited match in required skills")

        if scoring.experience_match >= 0.8:
            reason_parts.append("experience level aligns well with requirements")
        elif scoring.experience_match < 0.5:
            reason_parts.append("experience level below requirements")

        reason = ". ".join(reason_parts) + "."

        # Identify improvement areas
        improvement_areas = []
        if scoring.must_have_match < 0.8:
            improvement_areas.append("Acquire missing must-have skills")
        if scoring.project_relevance < 0.6:
            improvement_areas.append("Build projects with relevant technologies")
        if scoring.recency_factor < 0.7:
            improvement_areas.append("Gain more recent experience in target stack")

        return ExplainabilityReport(
            reason_for_score=reason,
            matched_strengths=matched_strengths[:5],
            missing_skills=missing_skills,
            improvement_areas=improvement_areas,
            recommendation=recommendation,
        )
