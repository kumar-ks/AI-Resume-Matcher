"""
Stage 5: Explainability
========================

PURPOSE:
    Generates human-readable reasoning for the match score.
    Provides reason for score, matched strengths, missing skills,
    improvement areas, and a recommendation.

CALL HIERARCHY:
    Called by:
        - pipeline.py → calls ExplainabilityEngine.explain() as Stage 5 of the pipeline
    Calls internally:
        - litellm.acompletion() → sends match data to LLM for natural language explanation
        - _build_prompt() → formats the explainability prompt with match data
        - _extract_json() → parses raw LLM response text into a Python dict
        - _try_parse_json() → attempts JSON parsing with fixups for common LLM issues
        - _parse_response() → converts parsed dict into ExplainabilityReport model
        - _fallback_explanation() → rule-based explanation when LLM fails

RETURNS:
    ExplainabilityReport with:
        - reason_for_score: 2-3 sentence explanation of the score
        - matched_strengths: list of key strengths matching the JD
        - missing_skills: list of skills/requirements the candidate lacks
        - improvement_areas: list of areas for improvement
        - recommendation: hiring recommendation string

FLOW:
    1. Receive JD, resume, scoring breakdown, and semantic result from pipeline.py
    2. Build prompt with all match data formatted for the LLM
    3. Call LLM via litellm.acompletion()
    4. Parse LLM response JSON (with multiple fallback strategies)
    5. Map parsed data to ExplainabilityReport model
    6. On failure, generate rule-based explanation from scoring data
    7. Return ExplainabilityReport to pipeline.py
"""

import logging
from typing import Optional

import litellm

from matching_engine.models import (
    ExplainabilityReport,
    JobDescription,
    ResumeProfile,
    ScoringBreakdown,
    SemanticMatchResult,
)
from matching_engine.utils import extract_json_from_llm_response

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
# This prompt instructs the LLM to act as an expert recruiter and generate
# a professional explanation of the match result. It receives all scoring data
# and must return structured JSON with reasoning and recommendations.
EXPLAINABILITY_PROMPT = """You are an expert recruiter providing feedback on a candidate match.

Given the following match data, generate a clear, professional explanation.

Job Title: {job_title}
Candidate: {candidate_name}
Qualification Score: {score}%

Must-Have Skills Required: {must_have_skills}
Candidate Skills: {candidate_skills}
Experience Required: {exp_required}
Candidate Experience: {candidate_exp} years

Scoring Breakdown:
- Must-have match: {must_have_score:.0%}
- Experience match: {exp_score:.0%}
- Skills depth: {skills_depth:.0%}
- Project relevance: {project_score:.0%}
- Recency factor: {recency_score:.0%}

Return a JSON object:
{{
    "reason_for_score": "2-3 sentence explanation of why this score was given",
    "matched_strengths": ["list of 3-5 key strengths that match the JD"],
    "missing_skills": ["list of skills/requirements the candidate lacks"],
    "improvement_areas": ["list of areas where candidate could improve for this role"],
    "recommendation": "One of: Strong Fit - Recommended for next round | Good Fit - Consider for interview | Partial Fit - May need additional screening | Weak Fit - Does not meet key requirements"
}}

Return ONLY valid JSON."""


class ExplainabilityEngine:
    """
    Generates human-readable explanations for match scores.

    Uses LLM to produce natural language reasoning about why a candidate
    received their score, what their strengths are, and what's missing.

    This is the entry point for Stage 5 of the matching pipeline.
    pipeline.py instantiates this class and calls explain() with all match data.
    """

    def __init__(self, model: str = "ollama/llama2", temperature: float = 0.3):
        """
        Initialize Explainability Engine.

        Called by: pipeline.py during pipeline initialization.

        Args:
            model: LiteLLM model identifier (e.g., "ollama/llama2", "gpt-4")
            temperature: LLM temperature (slightly higher than extraction for creativity)
        """
        self.model = model
        self.temperature = temperature
        logger.debug(f"ExplainabilityEngine initialized with model={model}, temperature={temperature}")

    async def explain(
        self,
        jd: JobDescription,
        resume: ResumeProfile,
        scoring: ScoringBreakdown,
        semantic: SemanticMatchResult,
    ) -> ExplainabilityReport:
        """
        Generate explanation for the match result.

        Called by: pipeline.py as Stage 5 of the matching pipeline.
        Calls: _build_prompt(), litellm.acompletion(), _extract_json(),
               _parse_response(), _fallback_explanation()

        Flow:
            1. Build the explainability prompt with all match data
            2. Call LLM via litellm.acompletion()
            3. Extract JSON from LLM response
            4. Parse JSON into ExplainabilityReport model
            5. On failure, generate rule-based fallback explanation

        Args:
            jd: Parsed job description (from Stage 1)
            resume: Parsed resume profile (from Stage 2)
            scoring: Scoring breakdown from Stage 4
            semantic: Semantic match result from Stage 3

        Returns:
            ExplainabilityReport with reasoning and recommendations
        """
        logger.info(f"Stage 5: Generating explanation for {resume.full_name}")
        logger.debug(f"Qualification score: {scoring.qualification_percentage}%")

        try:
            # ── Step 1: Build prompt with match data ──
            prompt_content = self._build_prompt(jd, resume, scoring)
            logger.debug(f"Built explainability prompt, length: {len(prompt_content)}")

            # ── Step 2: Call LLM for natural language explanation ──
            logger.debug(f"Sending explainability request to model: {self.model}")
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
                temperature=self.temperature,
                timeout=300,  # 5 min — matches other stages for Ollama cold start
            )

            # ── Step 3: Extract and parse JSON from response ──
            content = response.choices[0].message.content
            logger.debug(f"LLM response received, content length: {len(content) if content else 0}")

            data = extract_json_from_llm_response(content)
            if data is None:
                logger.warning("Could not extract JSON from explainability response, using fallback")
                return self._fallback_explanation(jd, resume, scoring)

            # ── Step 4: Convert to ExplainabilityReport model ──
            logger.debug(f"Successfully parsed explainability JSON with keys: {list(data.keys())}")
            return self._parse_response(data)

        except Exception as e:
            logger.error(f"Explainability generation failed: {e}")
            return self._fallback_explanation(jd, resume, scoring)

    def _build_prompt(
        self,
        jd: JobDescription,
        resume: ResumeProfile,
        scoring: ScoringBreakdown,
    ) -> str:
        """
        Build the explainability prompt with match data.

        Called by: explain()

        Formats the EXPLAINABILITY_PROMPT template with all relevant match data
        including job title, candidate name, scores, skills, and experience info.

        Args:
            jd: Parsed job description
            resume: Parsed resume profile
            scoring: Scoring breakdown from Stage 4

        Returns:
            Formatted prompt string ready to send to LLM
        """
        # Format must-have skills as comma-separated string
        must_have_skills = ", ".join(s.name for s in jd.must_have_skills) or "Not specified"

        # Limit candidate skills to first 15 to keep prompt concise
        candidate_skills = ", ".join(resume.skills[:15]) or "Not extracted"

        # Format experience requirement range
        exp_min = jd.experience_range_min or 0
        exp_max = jd.experience_range_max or "Not specified"
        exp_required = f"{exp_min}-{exp_max} years" if exp_max != "Not specified" else "Not specified"

        logger.debug(
            f"Building prompt: job='{jd.title}', candidate='{resume.full_name}', "
            f"score={scoring.qualification_percentage}%"
        )

        return EXPLAINABILITY_PROMPT.format(
            job_title=jd.title or "Not specified",
            candidate_name=resume.full_name or "Unknown",
            score=scoring.qualification_percentage,
            must_have_skills=must_have_skills,
            candidate_skills=candidate_skills,
            exp_required=exp_required,
            candidate_exp=resume.total_experience_years or "Unknown",
            must_have_score=scoring.must_have_match,
            exp_score=scoring.experience_match,
            skills_depth=scoring.skills_depth,
            project_score=scoring.project_relevance,
            recency_score=scoring.recency_factor,
        )

    def _parse_response(self, data: dict) -> ExplainabilityReport:
        """
        Parse LLM response into ExplainabilityReport.

        Called by: explain()

        Maps the JSON dict from the LLM into the ExplainabilityReport model.
        Uses .get() with defaults to handle missing fields gracefully.

        Args:
            data: Parsed JSON dict from LLM response

        Returns:
            ExplainabilityReport model
        """
        logger.debug(f"Parsing explainability response: recommendation='{data.get('recommendation', '')}'")
        return ExplainabilityReport(
            reason_for_score=data.get("reason_for_score", ""),
            matched_strengths=data.get("matched_strengths", []),
            missing_skills=data.get("missing_skills", []),
            improvement_areas=data.get("improvement_areas", []),
            recommendation=data.get("recommendation", ""),
        )

    def _fallback_explanation(
        self,
        jd: JobDescription,
        resume: ResumeProfile,
        scoring: ScoringBreakdown,
    ) -> ExplainabilityReport:
        """
        Generate a rule-based explanation when LLM is unavailable.

        Called by: explain() when LLM call or JSON parsing fails.

        Uses scoring breakdown to produce deterministic explanations.
        This ensures the pipeline always returns useful output even when
        the LLM is down or returns unparseable responses.

        Logic:
            1. Compare candidate skills against must-have skills for strengths/gaps
            2. Map qualification percentage to recommendation tier
            3. Build reason string from scoring dimension analysis
            4. Identify improvement areas from low-scoring dimensions

        Args:
            jd: Parsed job description
            resume: Parsed resume profile
            scoring: Scoring breakdown from Stage 4

        Returns:
            Rule-based ExplainabilityReport
        """
        logger.info("Using fallback rule-based explanation")
        logger.debug(
            f"Fallback triggered for {resume.full_name}, "
            f"score={scoring.qualification_percentage}%"
        )

        # ── Identify matched strengths and missing skills ──
        candidate_skills_lower = {s.lower() for s in resume.skills}
        matched_strengths = []
        missing_skills = []

        for skill in jd.must_have_skills:
            if skill.name.lower() in candidate_skills_lower:
                matched_strengths.append(skill.name)
            else:
                missing_skills.append(skill.name)

        logger.debug(f"Fallback: {len(matched_strengths)} strengths, {len(missing_skills)} missing")

        # ── Map score to recommendation tier ──
        score = scoring.qualification_percentage
        if score >= 85:
            recommendation = "Strong Fit - Recommended for next round."
        elif score >= 70:
            recommendation = "Good Fit - Consider for interview."
        elif score >= 50:
            recommendation = "Partial Fit - May need additional screening."
        else:
            recommendation = "Weak Fit - Does not meet key requirements."

        logger.debug(f"Fallback recommendation: '{recommendation}'")

        # ── Build reason string from scoring dimensions ──
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

        # ── Identify improvement areas from low-scoring dimensions ──
        improvement_areas = []
        if scoring.must_have_match < 0.8:
            improvement_areas.append("Acquire missing must-have skills")
        if scoring.project_relevance < 0.6:
            improvement_areas.append("Build projects with relevant technologies")
        if scoring.recency_factor < 0.7:
            improvement_areas.append("Gain more recent experience in target stack")

        logger.debug(f"Fallback improvement areas: {improvement_areas}")

        return ExplainabilityReport(
            reason_for_score=reason,
            matched_strengths=matched_strengths[:5],
            missing_skills=missing_skills,
            improvement_areas=improvement_areas,
            recommendation=recommendation,
        )
