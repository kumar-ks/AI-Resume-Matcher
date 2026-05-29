"""
AI Matching Engine Pipeline
============================

Orchestrates the 6-stage matching pipeline that evaluates candidates
against a job description.

PIPELINE STAGES:
    Stage 1: JD Understanding      → jd_understanding.py    → Extract requirements from JD
    Stage 2: Resume Understanding   → resume_understanding.py → Parse candidate profiles
    Stage 3: Semantic Matching      → semantic_matching.py   → Embedding-based similarity
    Stage 4: Scoring                → scoring.py             → Weighted qualification %
    Stage 5: Explainability         → explainability.py      → Human-readable reasoning
    Stage 6: Output                 → (this module)          → Assemble final MatchResult

CALL HIERARCHY:
    run.py → MatchingPipeline.__init__()     # Creates all stage engines
    run.py → pipeline.match(jd_text, resume_texts)
        → Stage 1: self.jd_understanding.extract(jd_text)
        → For each resume:
            → self._process_single_resume(jd, resume_text)
                → Stage 2: self.resume_understanding.extract(resume_text)
                → Stage 3: self.semantic_matcher.match(jd, resume)
                → Stage 4: self.scorer.score(jd, resume, semantic_result)
                → Stage 5: self.explainability.explain(jd, resume, scoring, semantic)
                → Stage 6: Assemble MatchResult
        → Sort results by qualification_percentage (descending)
        → Return sorted list

USAGE:
    pipeline = MatchingPipeline(model="ollama/llama3")
    results = await pipeline.match(jd_text, [resume_text_1, resume_text_2])
    for result in results:
        print(f"{result.candidate.full_name}: {result.qualification_percentage}%")
"""

import logging
from typing import Optional

from matching_engine.explainability import ExplainabilityEngine
from matching_engine.jd_understanding import JDUnderstanding
from matching_engine.models import (
    JobDescription,
    MatchResult,
    ResumeProfile,
)
from matching_engine.resume_understanding import ResumeUnderstanding
from matching_engine.scoring import Scorer
from matching_engine.semantic_matching import SemanticMatcher

logger = logging.getLogger(__name__)


class MatchingPipeline:
    """
    Orchestrates the full AI Matching Engine pipeline.

    Runs all 6 stages sequentially for each candidate and returns
    ranked results sorted by qualification percentage.
    """

    def __init__(
        self,
        model: str = "ollama/llama2",
        embedding_model: str = "all-MiniLM-L6-v2",
        scoring_weights: Optional[dict[str, float]] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize the matching pipeline with all stage engines.

        Called by: run.py → run_matching()

        Creates instances of:
            - JDUnderstanding (Stage 1): LLM-based JD parsing
            - ResumeUnderstanding (Stage 2): LLM + regex resume parsing
            - SemanticMatcher (Stage 3): Embedding model for similarity
            - Scorer (Stage 4): Weighted scoring formula
            - ExplainabilityEngine (Stage 5): LLM-based explanation generation

        Args:
            model: LiteLLM model identifier for LLM stages (e.g., "ollama/llama3")
            embedding_model: Sentence-transformers model for semantic matching
            scoring_weights: Custom scoring weights dict (optional, defaults to standard)
            temperature: LLM temperature for extraction stages (lower = more deterministic)
        """
        logger.debug(f"Initializing pipeline: model={model}, embeddings={embedding_model}")

        # Stage 1 engine: Extracts structured requirements from JD text
        self.jd_understanding = JDUnderstanding(model=model, temperature=temperature)

        # Stage 2 engine: Extracts structured profile from resume text
        self.resume_understanding = ResumeUnderstanding(model=model, temperature=temperature)

        # Stage 3 engine: Computes embedding-based similarity scores
        # Note: This pre-loads the embedding model (may take a few seconds on first run)
        self.semantic_matcher = SemanticMatcher(embedding_model=embedding_model)

        # Stage 4 engine: Computes weighted qualification percentage
        self.scorer = Scorer(weights=scoring_weights)

        # Stage 5 engine: Generates human-readable match explanations
        self.explainability = ExplainabilityEngine(model=model, temperature=0.3)

        logger.info(
            f"MatchingPipeline initialized (LLM: {model}, "
            f"Embeddings: {embedding_model})"
        )

    async def match(
        self,
        jd_text: str,
        resume_texts: list[str],
        jd: Optional[JobDescription] = None,
    ) -> list[MatchResult]:
        """
        Run the full matching pipeline for multiple candidates.

        Called by: run.py → run_matching()
        Calls: self.jd_understanding.extract(), self._process_single_resume()

        Args:
            jd_text: Raw job description text
            resume_texts: List of raw resume texts (one per candidate)
            jd: Pre-parsed JobDescription (skips Stage 1 if provided)

        Returns:
            List of MatchResult sorted by qualification_percentage (highest first)
        """
        logger.info(f"Starting pipeline: 1 JD x {len(resume_texts)} resumes")

        # ── Stage 1: JD Understanding ────────────────────────────────────────
        # Parse the job description into structured requirements
        if jd is None:
            jd = await self.jd_understanding.extract(jd_text)
        logger.info(f"JD parsed: {jd.title} ({len(jd.must_have_skills)} must-have skills)")
        logger.debug(f"  Must-have skills: {[s.name for s in jd.must_have_skills]}")
        logger.debug(f"  Good-to-have skills: {[s.name for s in jd.good_to_have_skills]}")
        logger.debug(f"  Experience range: {jd.experience_range_min}-{jd.experience_range_max} years")

        # ── Stages 2-6: Process each resume ──────────────────────────────────
        results: list[MatchResult] = []
        for i, resume_text in enumerate(resume_texts, 1):
            logger.info(f"Processing resume {i}/{len(resume_texts)}")
            result = await self._process_single_resume(jd, resume_text)
            results.append(result)

        # ── Sort results by score (highest first) ────────────────────────────
        results.sort(key=lambda r: r.qualification_percentage, reverse=True)

        if results:
            logger.info(
                f"Pipeline complete. Top match: "
                f"{results[0].candidate.full_name} ({results[0].qualification_percentage}%)"
            )

        return results

    async def match_single(
        self,
        jd_text: str,
        resume_text: str,
        jd: Optional[JobDescription] = None,
    ) -> MatchResult:
        """
        Run the pipeline for a single candidate (convenience method).

        Called by: External callers who want to match one resume at a time.
        Calls: self.match()
        """
        results = await self.match(jd_text, [resume_text], jd=jd)
        return results[0]

    async def _process_single_resume(
        self, jd: JobDescription, resume_text: str
    ) -> MatchResult:
        """
        Run stages 2-6 for a single resume against the parsed JD.

        Called by: self.match() for each resume in the batch
        Calls:
            → Stage 2: self.resume_understanding.extract(resume_text)
            → Stage 3: self.semantic_matcher.match(jd, resume)
            → Stage 4: self.scorer.score(jd, resume, semantic_result)
            → Stage 5: self.explainability.explain(jd, resume, scoring, semantic)
            → Stage 6: Assemble MatchResult

        Args:
            jd: Parsed JobDescription from Stage 1
            resume_text: Raw resume text for this candidate

        Returns:
            MatchResult with all scores, explanations, and recommendations
        """
        # ── Stage 2: Resume Understanding ────────────────────────────────────
        # Extract structured profile (name, skills, experience, etc.)
        resume = await self.resume_understanding.extract(resume_text)
        logger.info(f"  Resume parsed: {resume.full_name}")
        logger.debug(f"    Skills: {len(resume.skills)} items")
        logger.debug(f"    Experience: {resume.total_experience_years} years")
        logger.debug(f"    Work history: {len(resume.work_experiences)} entries")

        # ── Stage 3: Semantic Matching ───────────────────────────────────────
        # Compute embedding-based similarity across multiple dimensions
        semantic_result = self.semantic_matcher.match(jd, resume)
        logger.info(
            f"  Semantic scores: skill={semantic_result.skill_relevance:.2f}, "
            f"role={semantic_result.role_alignment:.2f}"
        )
        logger.debug(
            f"    Full semantic: contextual={semantic_result.contextual_similarity:.2f}, "
            f"domain={semantic_result.domain_relevance:.2f}, "
            f"tech={semantic_result.technology_mapping:.2f}, "
            f"exp={semantic_result.experience_alignment:.2f}"
        )

        # ── Stage 4: Scoring ─────────────────────────────────────────────────
        # Calculate weighted qualification percentage
        scoring = self.scorer.score(jd, resume, semantic_result)
        logger.info(f"  Score: {scoring.qualification_percentage}%")
        logger.debug(
            f"    Breakdown: must_have={scoring.must_have_match:.2f}, "
            f"exp={scoring.experience_match:.2f}, "
            f"depth={scoring.skills_depth:.2f}, "
            f"project={scoring.project_relevance:.2f}, "
            f"recency={scoring.recency_factor:.2f}"
        )

        # ── Stage 5: Explainability ──────────────────────────────────────────
        # Generate human-readable reasoning for the score
        explanation = await self.explainability.explain(
            jd, resume, scoring, semantic_result
        )
        logger.debug(f"    Recommendation: {explanation.recommendation}")

        # ── Stage 6: Output — Assemble final MatchResult ─────────────────────
        return MatchResult(
            candidate=resume,
            job_description=jd,
            qualification_percentage=scoring.qualification_percentage,
            semantic_scores=semantic_result,
            scoring_breakdown=scoring,
            explainability=explanation,
            key_strengths=explanation.matched_strengths,
            missing_skills=explanation.missing_skills,
            reasoning=explanation.reason_for_score,
            recommendation=explanation.recommendation,
        )

    async def match_with_parsed_inputs(
        self,
        jd: JobDescription,
        resumes: list[ResumeProfile],
    ) -> list[MatchResult]:
        """
        Run stages 3-6 with pre-parsed JD and resumes (skips file loading and LLM extraction).

        Useful when JD and resumes have already been extracted
        (e.g., from a database or previous pipeline run).

        Called by: External callers with pre-parsed data
        Calls: Stages 3-6 directly
        """
        logger.info(f"Running stages 3-6 for {len(resumes)} pre-parsed resumes")

        results: list[MatchResult] = []
        for resume in resumes:
            semantic_result = self.semantic_matcher.match(jd, resume)
            scoring = self.scorer.score(jd, resume, semantic_result)
            explanation = await self.explainability.explain(
                jd, resume, scoring, semantic_result
            )

            result = MatchResult(
                candidate=resume,
                job_description=jd,
                qualification_percentage=scoring.qualification_percentage,
                semantic_scores=semantic_result,
                scoring_breakdown=scoring,
                explainability=explanation,
                key_strengths=explanation.matched_strengths,
                missing_skills=explanation.missing_skills,
                reasoning=explanation.reason_for_score,
                recommendation=explanation.recommendation,
            )
            results.append(result)

        results.sort(key=lambda r: r.qualification_percentage, reverse=True)
        return results
