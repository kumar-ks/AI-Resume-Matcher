"""
Stage 4: Scoring
=================

PURPOSE:
    Weighted scoring model that computes qualification percentage.
    Evaluates must-have match, experience match, skills depth,
    project relevance, and recency factor.

CALL HIERARCHY:
    Called by:
        - pipeline.py → calls Scorer.score() as Stage 4 of the matching pipeline
    Calls internally:
        - _score_must_have_match() → scores how many must-have skills candidate has
        - _score_experience_match() → scores experience level alignment
        - _score_skills_depth() → scores depth of skills (breadth + semantic)
        - _score_project_relevance() → scores relevance of candidate's projects
        - _score_recency() → scores recency of relevant experience
        - _fuzzy_skill_match() → fuzzy matching utility for skill name variations

RETURNS:
    ScoringBreakdown with:
        - Individual dimension scores (0.0 to 1.0 each)
        - Final qualification_percentage (0-100%, weighted sum of dimensions)

FLOW:
    1. Receive parsed JD, resume, and SemanticMatchResult from pipeline.py
    2. Compute 5 scoring dimensions independently
    3. Apply configurable weights to each dimension
    4. Compute weighted sum as final qualification percentage
    5. Return ScoringBreakdown to pipeline.py for use in Stage 5 (Explainability)
"""

import logging
from datetime import datetime

from matching_engine.models import (
    JobDescription,
    ResumeProfile,
    ScoringBreakdown,
    SemanticMatchResult,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SCORING WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# These weights determine how much each dimension contributes to the final score.
# They must sum to 1.0. The heaviest weight is on must-have skills (0.35),
# reflecting that core skill match is the most important hiring signal.
DEFAULT_WEIGHTS = {
    "must_have_match": 0.35,      # Core required skills coverage
    "experience_match": 0.25,     # Years/level alignment
    "skills_depth": 0.20,         # Depth beyond just having skills
    "project_relevance": 0.12,    # Relevant project experience
    "recency_factor": 0.08,       # How recent the experience is
}


class Scorer:
    """
    Computes weighted qualification score for a candidate against a JD.

    Combines multiple scoring dimensions with configurable weights
    to produce a final qualification percentage (0-100%).

    This is the entry point for Stage 4 of the matching pipeline.
    pipeline.py instantiates this class and calls score() with JD, resume,
    and the SemanticMatchResult from Stage 3.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        """
        Initialize Scorer with optional custom weights.

        Called by: pipeline.py during pipeline initialization.

        Args:
            weights: Dict mapping dimension names to weights (must sum to 1.0).
                     Uses DEFAULT_WEIGHTS if not provided.
        """
        self.weights = weights or DEFAULT_WEIGHTS

        # Validate and normalize weights to ensure they sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Scoring weights sum to {total}, normalizing to 1.0")
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.debug(f"Scorer initialized with weights: {self.weights}")

    def score(
        self,
        jd: JobDescription,
        resume: ResumeProfile,
        semantic_result: SemanticMatchResult,
    ) -> ScoringBreakdown:
        """
        Compute scoring breakdown for a candidate.

        Called by: pipeline.py as Stage 4 of the matching pipeline.
        Calls: _score_must_have_match(), _score_experience_match(),
               _score_skills_depth(), _score_project_relevance(), _score_recency()

        Flow:
            1. Compute each of the 5 scoring dimensions
            2. Apply weights to each dimension
            3. Sum weighted scores and multiply by 100 for percentage
            4. Return ScoringBreakdown with all scores

        Args:
            jd: Parsed job description (from Stage 1)
            resume: Parsed resume profile (from Stage 2)
            semantic_result: Output from Stage 3 (semantic matching)

        Returns:
            ScoringBreakdown with individual scores and final percentage
        """
        logger.info(f"Stage 4: Scoring candidate {resume.full_name}")

        # ── Compute each scoring dimension independently ──
        must_have_match = self._score_must_have_match(jd, resume)
        logger.debug(f"Must-have match score: {must_have_match:.3f}")

        experience_match = self._score_experience_match(jd, resume)
        logger.debug(f"Experience match score: {experience_match:.3f}")

        skills_depth = self._score_skills_depth(jd, resume, semantic_result)
        logger.debug(f"Skills depth score: {skills_depth:.3f}")

        project_relevance = self._score_project_relevance(jd, resume)
        logger.debug(f"Project relevance score: {project_relevance:.3f}")

        recency_factor = self._score_recency(resume)
        logger.debug(f"Recency factor score: {recency_factor:.3f}")

        # ── Compute weighted sum for final qualification percentage ──
        qualification_percentage = (
            must_have_match * self.weights["must_have_match"]
            + experience_match * self.weights["experience_match"]
            + skills_depth * self.weights["skills_depth"]
            + project_relevance * self.weights["project_relevance"]
            + recency_factor * self.weights["recency_factor"]
        ) * 100

        logger.debug(f"Final qualification percentage: {qualification_percentage:.1f}%")

        return ScoringBreakdown(
            must_have_match=must_have_match,
            experience_match=experience_match,
            skills_depth=skills_depth,
            project_relevance=project_relevance,
            recency_factor=recency_factor,
            qualification_percentage=round(qualification_percentage, 1),
        )

    def _score_must_have_match(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        Score how many must-have skills the candidate possesses.

        Called by: score()
        Calls: _fuzzy_skill_match()

        Uses fuzzy matching to account for skill name variations
        (e.g., "kubernetes" matches "k8s", "React" matches "ReactJS").

        Gathers candidate skills from:
            - resume.skills (explicit skill list)
            - work_experiences[].technologies
            - projects[].technologies

        Returns:
            Float 0.0-1.0 (ratio of matched must-have skills)
        """
        # If JD has no must-have skills, candidate gets full score
        if not jd.must_have_skills:
            logger.debug("No must-have skills in JD, returning 1.0")
            return 1.0

        # Build comprehensive set of candidate skills (lowercased for matching)
        candidate_skills_lower = {s.lower() for s in resume.skills}
        for exp in resume.work_experiences:
            candidate_skills_lower.update(t.lower() for t in exp.technologies)
        for proj in resume.projects:
            candidate_skills_lower.update(t.lower() for t in proj.technologies)

        logger.debug(f"Candidate has {len(candidate_skills_lower)} unique skills (lowercased)")

        # Count how many must-have skills the candidate matches
        matched = 0
        for skill in jd.must_have_skills:
            skill_name = skill.name.lower()
            if self._fuzzy_skill_match(skill_name, candidate_skills_lower):
                matched += 1
                logger.debug(f"  Must-have MATCHED: '{skill.name}'")
            else:
                logger.debug(f"  Must-have MISSING: '{skill.name}'")

        score = matched / len(jd.must_have_skills)
        logger.debug(f"Must-have match: {matched}/{len(jd.must_have_skills)} = {score:.3f}")
        return score

    def _score_experience_match(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        Score experience level alignment.

        Called by: score()

        Scoring logic:
            - If JD doesn't specify experience: return 0.8 (slightly positive)
            - If candidate experience unknown: return 0.5 (neutral)
            - If within range: return 1.0 (perfect)
            - If under-experienced: penalize based on gap (min 0.2)
            - If over-experienced: slight penalty (min 0.6)

        Returns:
            Float 0.2-1.0 representing experience alignment
        """
        # No experience requirement in JD → slightly positive default
        if jd.experience_range_min is None and jd.experience_range_max is None:
            logger.debug("No experience requirement in JD, returning 0.8")
            return 0.8

        candidate_years = resume.total_experience_years
        if candidate_years is None:
            logger.debug("Candidate experience years unknown, returning 0.5")
            return 0.5

        min_years = jd.experience_range_min or 0
        max_years = jd.experience_range_max or min_years + 5

        logger.debug(f"Experience: candidate={candidate_years}yrs, required={min_years}-{max_years}yrs")

        if min_years <= candidate_years <= max_years:
            # Perfect match
            return 1.0
        elif candidate_years < min_years:
            # Under-experienced: penalize proportionally to gap
            gap = min_years - candidate_years
            score = max(0.2, 1.0 - (gap / max(min_years, 1)) * 0.8)
            logger.debug(f"Under-experienced by {gap}yrs, score={score:.3f}")
            return score
        else:
            # Over-experienced: slight penalty (still valuable)
            excess = candidate_years - max_years
            score = max(0.6, 1.0 - excess * 0.03)
            logger.debug(f"Over-experienced by {excess}yrs, score={score:.3f}")
            return score

    def _score_skills_depth(
        self,
        jd: JobDescription,
        resume: ResumeProfile,
        semantic_result: SemanticMatchResult,
    ) -> float:
        """
        Score depth of skills beyond just having them.

        Called by: score()
        Calls: _fuzzy_skill_match()

        Considers semantic skill relevance (from Stage 3), number of related
        projects, and duration of experience with key technologies.

        Combines:
            - breadth_score (60%): ratio of all JD skills matched
            - semantic skill_relevance (40%): from Stage 3 embedding comparison

        Returns:
            Float 0.0-1.0 representing skills depth
        """
        all_jd_skills = jd.must_have_skills + jd.good_to_have_skills
        if not all_jd_skills:
            logger.debug("No JD skills defined, returning 0.8")
            return 0.8

        # Build candidate skills set from resume + work experiences
        candidate_skills_lower = {s.lower() for s in resume.skills}
        for exp in resume.work_experiences:
            candidate_skills_lower.update(t.lower() for t in exp.technologies)

        # Count how many total JD skills (must-have + good-to-have) are matched
        total_skills = len(all_jd_skills)
        matched = sum(
            1 for s in all_jd_skills
            if self._fuzzy_skill_match(s.name.lower(), candidate_skills_lower)
        )

        # Breadth score: simple ratio of matched skills
        breadth_score = matched / total_skills

        # Depth score: blend breadth (60%) with semantic relevance from Stage 3 (40%)
        depth_score = (breadth_score * 0.6) + (semantic_result.skill_relevance * 0.4)
        result = min(1.0, depth_score)

        logger.debug(
            f"Skills depth: matched={matched}/{total_skills}, "
            f"breadth={breadth_score:.3f}, semantic={semantic_result.skill_relevance:.3f}, "
            f"final={result:.3f}"
        )
        return result

    def _score_project_relevance(
        self, jd: JobDescription, resume: ResumeProfile
    ) -> float:
        """
        Score relevance of candidate's projects to the JD.

        Called by: score()

        Considers technology overlap between JD skills and project technologies.
        For each project, computes overlap ratio with JD tech requirements.
        Final score blends best project (60%) with average across all projects (40%).

        Returns:
            Float 0.0-1.0 representing project relevance
        """
        # No projects listed → use work experience as partial signal
        if not resume.projects:
            if resume.work_experiences:
                logger.debug("No projects but has work experience, returning 0.6")
                return 0.6
            logger.debug("No projects and no work experience, returning 0.3")
            return 0.3

        # Gather all JD technologies (lowercased)
        jd_techs = {s.name.lower() for s in jd.must_have_skills + jd.good_to_have_skills}
        if not jd_techs:
            logger.debug("No JD techs to compare against, returning 0.7")
            return 0.7

        # Score each project by technology overlap with JD requirements
        project_scores = []
        for project in resume.projects:
            proj_techs = {t.lower() for t in project.technologies}
            if proj_techs and jd_techs:
                overlap = len(proj_techs & jd_techs) / len(jd_techs)
                project_scores.append(overlap)

        if not project_scores:
            logger.debug("No project tech overlap found, returning 0.4")
            return 0.4

        # Blend best project score (60%) with average (40%)
        best_score = max(project_scores)
        avg_score = sum(project_scores) / len(project_scores)
        result = (best_score * 0.6) + (avg_score * 0.4)

        logger.debug(
            f"Project relevance: {len(project_scores)} projects scored, "
            f"best={best_score:.3f}, avg={avg_score:.3f}, final={result:.3f}"
        )
        return result

    def _score_recency(self, resume: ResumeProfile) -> float:
        """
        Score recency of relevant experience.

        Called by: score()

        More recent experience with relevant technologies scores higher.
        Current positions get 1.0, past positions decay by 0.1 per year since end.
        Final score blends most recent (60%) with average of remaining (40%).

        Returns:
            Float 0.3-1.0 representing recency of experience
        """
        if not resume.work_experiences:
            logger.debug("No work experiences, returning 0.5")
            return 0.5

        current_year = datetime.now().year
        recency_scores = []

        for exp in resume.work_experiences:
            if exp.is_current:
                # Current position gets maximum recency score
                recency_scores.append(1.0)
            elif exp.end_year:
                # Past positions decay based on how many years ago they ended
                years_ago = current_year - exp.end_year
                score = max(0.3, 1.0 - (years_ago * 0.1))
                recency_scores.append(score)

        if not recency_scores:
            logger.debug("No recency data available from experiences, returning 0.6")
            return 0.6

        # Sort descending to get most recent first
        recency_scores.sort(reverse=True)

        if len(recency_scores) == 1:
            logger.debug(f"Single experience recency score: {recency_scores[0]:.3f}")
            return recency_scores[0]

        # Blend: most recent (60%) + average of rest (40%)
        result = recency_scores[0] * 0.6 + (
            sum(recency_scores[1:]) / len(recency_scores[1:])
        ) * 0.4

        logger.debug(f"Recency: most_recent={recency_scores[0]:.3f}, blended={result:.3f}")
        return result

    def _fuzzy_skill_match(self, skill: str, candidate_skills: set[str]) -> bool:
        """
        Check if a skill matches any candidate skill with fuzzy logic.

        Called by: _score_must_have_match(), _score_skills_depth()

        Handles common variations like abbreviations, framework versions, etc.

        Matching strategies (tried in order):
            1. Exact match in candidate skills set
            2. Substring containment (either direction)
            3. Known abbreviation/alias lookup table

        Args:
            skill: Lowercased skill name to search for
            candidate_skills: Set of lowercased candidate skill names

        Returns:
            True if skill matches any candidate skill
        """
        # Strategy 1: Exact match
        if skill in candidate_skills:
            return True

        # Strategy 2: Substring containment (e.g., "react" in "reactjs")
        for cs in candidate_skills:
            if skill in cs or cs in skill:
                return True

        # Strategy 3: Known abbreviation/alias lookup
        # Maps canonical skill names to sets of known variations
        abbreviations = {
            "javascript": {"js", "javascript", "ecmascript"},
            "typescript": {"ts", "typescript"},
            "python": {"python", "py"},
            "kubernetes": {"kubernetes", "k8s"},
            "react": {"react", "reactjs", "react.js"},
            "node": {"node", "nodejs", "node.js"},
            "aws": {"aws", "amazon web services"},
            "gcp": {"gcp", "google cloud", "google cloud platform"},
            "azure": {"azure", "microsoft azure"},
            "docker": {"docker", "containerization"},
            "ci/cd": {"ci/cd", "cicd", "ci cd", "continuous integration"},
            "sql": {"sql", "mysql", "postgresql", "postgres"},
            "nosql": {"nosql", "mongodb", "dynamodb", "cassandra"},
            "microservices": {"microservices", "micro-services", "micro services"},
            "spring boot": {"spring boot", "springboot", "spring-boot"},
        }

        for canonical, variants in abbreviations.items():
            if skill in variants:
                if any(cs in variants or canonical in cs for cs in candidate_skills):
                    return True

        return False
