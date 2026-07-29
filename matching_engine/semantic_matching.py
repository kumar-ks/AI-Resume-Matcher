"""
Stage 3: Semantic Matching
===========================

PURPOSE:
    Compares resume against job description using embedding-based similarity.
    Evaluates contextual similarity, skill relevance, role alignment,
    domain relevance, technology mapping, and experience alignment.

CALL HIERARCHY:
    Called by:
        - pipeline.py → calls SemanticMatcher.match() as Stage 3 of the matching pipeline
    Calls internally:
        - SentenceTransformer (sentence-transformers library) → generates text embeddings
        - _compute_contextual_similarity() → overall JD vs resume text similarity
        - _compute_skill_relevance() → JD skills vs candidate skills similarity
        - _compute_role_alignment() → role/responsibilities alignment
        - _compute_domain_relevance() → domain/industry alignment
        - _compute_technology_mapping() → technology stack alignment
        - _compute_experience_alignment() → experience years/level alignment
        - _cosine_similarity() → core embedding comparison utility
        - _build_jd_text() → constructs text from structured JD fields

RETURNS:
    SemanticMatchResult with scores (0.0 to 1.0) for each dimension:
        contextual_similarity, skill_relevance, role_alignment,
        domain_relevance, technology_mapping, experience_alignment

FLOW:
    1. Receive parsed JD and resume from pipeline.py
    2. Compute 6 semantic similarity dimensions using embeddings
    3. Return SemanticMatchResult to pipeline.py for use in Stage 4 (Scoring)
"""

import logging
from typing import Optional

import numpy as np

from matching_engine.models import (
    JobDescription,
    ResumeProfile,
    SemanticMatchResult,
)

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """
    Performs semantic matching between a resume and job description.

    Uses sentence embeddings to compute similarity across multiple
    dimensions: skills, role, domain, technology, and experience.

    This is the entry point for Stage 3 of the matching pipeline.
    pipeline.py instantiates this class and calls match() with parsed JD and resume.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Semantic Matcher.

        Called by: pipeline.py during pipeline initialization.

        Args:
            embedding_model: Sentence-transformers model name for embeddings
        """
        self.embedding_model_name = embedding_model
        self._model = None
        # Pre-load the model immediately to avoid async/HTTP issues later
        logger.debug(f"SemanticMatcher initializing, pre-loading model: {embedding_model}")
        _ = self.model

    @property
    def model(self):
        """
        Lazy-load the sentence transformer model.

        Called by: __init__() for pre-loading, and _cosine_similarity() for encoding.

        Handles corporate proxy/SSL issues by temporarily patching httpx.Client
        to disable SSL verification during model download. After loading, sets
        HF_HUB_OFFLINE=1 to prevent further HTTP calls.

        Returns:
            Loaded SentenceTransformer model instance
        """
        if self._model is None:
            from matching_engine.embedding_cache import get_embedding_model
            self._model = get_embedding_model(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")

        return self._model

    def match(self, jd: JobDescription, resume: ResumeProfile) -> SemanticMatchResult:
        """
        Compute semantic similarity between JD and resume across dimensions.

        Called by: pipeline.py as Stage 3 of the matching pipeline.
        Calls: _compute_contextual_similarity(), _compute_skill_relevance(),
               _compute_role_alignment(), _compute_domain_relevance(),
               _compute_technology_mapping(), _compute_experience_alignment()

        Flow:
            1. Compute each of the 6 similarity dimensions independently
            2. Package all scores into SemanticMatchResult
            3. Return to pipeline.py (used by Stage 4 Scoring and Stage 5 Explainability)

        Args:
            jd: Parsed job description (from Stage 1)
            resume: Parsed resume profile (from Stage 2)

        Returns:
            SemanticMatchResult with scores for each dimension (0.0 to 1.0)
        """
        logger.info(f"Stage 3: Semantic matching for {resume.full_name}")

        # ── Compute each similarity dimension ──
        logger.debug("Computing contextual similarity (full text comparison)")
        contextual_similarity = self._compute_contextual_similarity(jd, resume)

        logger.debug("Computing skill relevance (skills list comparison)")
        skill_relevance = self._compute_skill_relevance(jd, resume)

        logger.debug("Computing role alignment (title + responsibilities)")
        role_alignment = self._compute_role_alignment(jd, resume)

        logger.debug("Computing domain relevance (industry/domain comparison)")
        domain_relevance = self._compute_domain_relevance(jd, resume)

        logger.debug("Computing technology mapping (tech stack comparison)")
        technology_mapping = self._compute_technology_mapping(jd, resume)

        logger.debug("Computing experience alignment (years + level)")
        experience_alignment = self._compute_experience_alignment(jd, resume)

        logger.debug(
            f"Semantic match scores: contextual={contextual_similarity:.3f}, "
            f"skill={skill_relevance:.3f}, role={role_alignment:.3f}, "
            f"domain={domain_relevance:.3f}, tech={technology_mapping:.3f}, "
            f"experience={experience_alignment:.3f}"
        )

        return SemanticMatchResult(
            contextual_similarity=contextual_similarity,
            skill_relevance=skill_relevance,
            role_alignment=role_alignment,
            domain_relevance=domain_relevance,
            technology_mapping=technology_mapping,
            experience_alignment=experience_alignment,
        )

    def _compute_contextual_similarity(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        Overall contextual similarity between full JD and resume text.

        Called by: match()
        Calls: _build_jd_text(), _cosine_similarity()

        Compares the entire JD text against the entire resume text to get
        a high-level semantic similarity score.

        Returns:
            Float 0.0-1.0 representing overall text similarity
        """
        # Use raw text if available, otherwise build from structured fields
        jd_text = jd.raw_text or self._build_jd_text(jd)
        resume_text = resume.raw_text or resume.career_summary

        if not jd_text or not resume_text:
            logger.debug("Contextual similarity: missing text, returning 0.0")
            return 0.0

        return self._cosine_similarity(jd_text, resume_text)

    def _compute_skill_relevance(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        Semantic similarity between required skills and candidate skills.

        Called by: match()
        Calls: _cosine_similarity()

        Joins all JD skills and all resume skills into comma-separated strings,
        then computes embedding similarity between them.

        Returns:
            Float 0.0-1.0 representing skill relevance
        """
        # Gather all JD skills (must-have + good-to-have)
        jd_skills = [s.name for s in jd.must_have_skills + jd.good_to_have_skills]
        resume_skills = resume.skills

        if not jd_skills or not resume_skills:
            logger.debug("Skill relevance: empty skill lists, returning 0.0")
            return 0.0

        jd_skills_text = ", ".join(jd_skills)
        resume_skills_text = ", ".join(resume_skills)
        logger.debug(f"Skill relevance: comparing {len(jd_skills)} JD skills vs {len(resume_skills)} resume skills")

        return self._cosine_similarity(jd_skills_text, resume_skills_text)

    def _compute_role_alignment(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        How well the candidate's role history aligns with the target role.

        Called by: match()
        Calls: _cosine_similarity()

        Compares JD title + responsibilities against candidate's work history
        (titles, companies, and responsibilities from all positions).

        Returns:
            Float 0.0-1.0 representing role alignment
        """
        # Build JD role text from title and responsibilities
        jd_role_text = f"{jd.title}. {' '.join(jd.responsibilities)}"

        # Build resume role text from all work experiences
        resume_role_text = " ".join(
            f"{exp.title} at {exp.company}. {' '.join(exp.responsibilities)}"
            for exp in resume.work_experiences
        )

        if not jd_role_text.strip() or not resume_role_text.strip():
            logger.debug("Role alignment: missing role text, returning 0.0")
            return 0.0

        return self._cosine_similarity(jd_role_text, resume_role_text)

    def _compute_domain_relevance(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        Domain/industry alignment between JD and candidate experience.

        Called by: match()
        Calls: _cosine_similarity()

        Compares domain/industry lists. If either is missing, falls back to
        comparing title vs career summary, or returns a neutral 0.5 score.

        Returns:
            Float 0.0-1.0 representing domain relevance
        """
        jd_domains = jd.domain_industry
        resume_domains = resume.domain_expertise

        if not jd_domains or not resume_domains:
            # Fallback: use title/career summary if domain lists are empty
            if jd.raw_text and resume.career_summary:
                logger.debug("Domain relevance: using fallback (title vs career summary)")
                return self._cosine_similarity(
                    " ".join(jd.domain_industry) if jd.domain_industry else jd.title,
                    " ".join(resume.domain_expertise) if resume.domain_expertise else resume.career_summary,
                )
            logger.debug("Domain relevance: no domain info available, returning neutral 0.5")
            return 0.5  # Neutral score when domain info is missing

        jd_domain_text = ", ".join(jd_domains)
        resume_domain_text = ", ".join(resume_domains)
        logger.debug(f"Domain relevance: comparing {len(jd_domains)} JD domains vs {len(resume_domains)} resume domains")

        return self._cosine_similarity(jd_domain_text, resume_domain_text)

    def _compute_technology_mapping(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        Technology stack alignment between JD requirements and resume.

        Called by: match()
        Calls: _cosine_similarity()

        Gathers all technologies from JD skills and from resume (skills +
        work experience technologies + project technologies), then compares
        them semantically.

        Returns:
            Float 0.0-1.0 representing technology alignment
        """
        # Gather JD technologies from all skill entries
        jd_techs = [s.name for s in jd.must_have_skills + jd.good_to_have_skills]

        # Gather resume technologies from skills, work experiences, and projects
        resume_techs = set(resume.skills)
        for exp in resume.work_experiences:
            resume_techs.update(exp.technologies)
        for proj in resume.projects:
            resume_techs.update(proj.technologies)

        if not jd_techs or not resume_techs:
            logger.debug("Technology mapping: empty tech lists, returning 0.0")
            return 0.0

        jd_tech_text = ", ".join(jd_techs)
        resume_tech_text = ", ".join(resume_techs)
        logger.debug(f"Technology mapping: {len(jd_techs)} JD techs vs {len(resume_techs)} resume techs")

        return self._cosine_similarity(jd_tech_text, resume_tech_text)

    def _compute_experience_alignment(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        How well the candidate's experience level matches JD requirements.

        Called by: match()

        Considers both years of experience and seniority level.
        Uses a rule-based approach (not embeddings) since experience is numeric.

        Scoring logic:
            - If JD doesn't specify experience: return 0.7 (neutral)
            - If candidate experience is unknown: return 0.5
            - If within range: return 1.0 (perfect match)
            - If under-experienced: linear decrease based on gap
            - If over-experienced: slight penalty (still valuable)

        Returns:
            Float 0.0-1.0 representing experience alignment
        """
        # If JD doesn't specify experience requirements, return neutral score
        if jd.experience_range_min is None and jd.experience_range_max is None:
            logger.debug("Experience alignment: JD has no experience requirement, returning 0.7")
            return 0.7  # Neutral when JD doesn't specify

        candidate_years = resume.total_experience_years
        if candidate_years is None:
            logger.debug("Experience alignment: candidate years unknown, returning 0.5")
            return 0.5  # Unknown experience

        min_years = jd.experience_range_min or 0
        max_years = jd.experience_range_max or min_years + 5

        logger.debug(
            f"Experience alignment: candidate={candidate_years}yrs, "
            f"required={min_years}-{max_years}yrs"
        )

        if min_years <= candidate_years <= max_years:
            # Perfect match - candidate is within the required range
            logger.debug("Experience alignment: perfect match (within range)")
            return 1.0
        elif candidate_years < min_years:
            # Under-experienced: score decreases linearly based on gap
            gap = min_years - candidate_years
            score = max(0.0, 1.0 - (gap / max(min_years, 1)))
            logger.debug(f"Experience alignment: under-experienced by {gap}yrs, score={score:.3f}")
            return score
        else:
            # Over-experienced: slight penalty (still valuable)
            excess = candidate_years - max_years
            score = max(0.5, 1.0 - (excess * 0.05))
            logger.debug(f"Experience alignment: over-experienced by {excess}yrs, score={score:.3f}")
            return score

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two text strings using embeddings.

        Called by: _compute_contextual_similarity(), _compute_skill_relevance(),
                   _compute_role_alignment(), _compute_domain_relevance(),
                   _compute_technology_mapping()

        Flow:
            1. Encode both texts into embedding vectors using SentenceTransformer
            2. Compute cosine similarity (dot product / product of norms)
            3. Clamp result to [0, 1] range
            4. On failure, attempt model reinitialization (handles closed HTTP client)

        Args:
            text_a: First text string
            text_b: Second text string

        Returns:
            Float 0.0-1.0 representing cosine similarity
        """
        try:
            # Encode both texts into dense vectors
            embeddings = self.model.encode([text_a, text_b], show_progress_bar=False)

            # Compute cosine similarity: dot(a, b) / (||a|| * ||b||)
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )

            # Clamp to [0, 1] range (cosine similarity can be negative for dissimilar texts)
            result = float(max(0.0, min(1.0, similarity)))
            logger.debug(f"Cosine similarity computed: {result:.4f}")
            return result
        except Exception as e:
            # If the model's HTTP client was closed, try reinitializing
            if "client has been closed" in str(e) or "Cannot send a request" in str(e):
                logger.warning("Embedding model HTTP client closed, reinitializing...")
                self._model = None
                try:
                    embeddings = self.model.encode([text_a, text_b], show_progress_bar=False)
                    similarity = np.dot(embeddings[0], embeddings[1]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                    )
                    return float(max(0.0, min(1.0, similarity)))
                except Exception as e2:
                    logger.error(f"Embedding computation failed after reinit: {e2}")
                    return 0.0
            logger.error(f"Embedding computation failed: {e}")
            return 0.0

    def _build_jd_text(self, jd: JobDescription) -> str:
        """
        Build a text representation from structured JD fields.

        Called by: _compute_contextual_similarity() when jd.raw_text is empty.

        Concatenates title, skill names, and responsibilities into a single
        period-separated string for embedding.

        Args:
            jd: Structured JobDescription model

        Returns:
            Concatenated text representation of the JD
        """
        parts = [jd.title]
        parts.extend(s.name for s in jd.must_have_skills)
        parts.extend(s.name for s in jd.good_to_have_skills)
        parts.extend(jd.responsibilities)
        text = ". ".join(parts)
        logger.debug(f"Built JD text from structured fields, length: {len(text)}")
        return text
