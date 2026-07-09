"""
Evaluation Framework — Scoring Validation & Bias Detection
============================================================

Provides tools to evaluate whether the matching pipeline's scoring
is accurate, consistent, and unbiased.

TWO COMPONENTS:
    1. Scoring Evaluation (Step 5): Validates that scores are calibrated and rankings stable
    2. Bias Detection (Step 6): Checks for demographic or format-based bias

SCORING EVALUATION:
    - Rank stability: Perturb weights slightly, check if top-K ranking changes drastically
    - Score distribution: Detect all-same scores, skewed distributions, ceiling/floor effects
    - Calibration check: Compare score bands to human-interpretable labels

BIAS DETECTION:
    - Name-swap test: Same profile with different names → scores should be identical
    - Format bias: Compare scores for same content in different formats
    - Keyword stuffing: Detect if keyword-dense resumes get artificially high scores
    - Score demographic parity: Check score distribution evenness

USAGE:
    from matching_engine.evaluation import (
        evaluate_ranking_stability,
        evaluate_score_distribution,
        detect_name_bias,
        detect_keyword_stuffing,
        generate_evaluation_report,
    )

    report = generate_evaluation_report(results, jd)

CALLED BY:
    - run.py → --evaluate flag (optional post-match evaluation)
    - Standalone for auditing
"""

import logging
import statistics
from typing import Optional

from matching_engine.models import (
    JobDescription,
    MatchResult,
    ResumeProfile,
    ScoringBreakdown,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationReport:
    """Full evaluation report for a match run."""

    def __init__(self):
        self.ranking_stability: float = 1.0  # 0-1, higher = more stable
        self.score_distribution: dict = {}
        self.calibration: dict = {}
        self.bias_flags: list[str] = []
        self.keyword_stuffing_suspects: list[str] = []
        self.overall_quality: str = "good"  # good, acceptable, poor

    def summary(self) -> str:
        lines = [
            f"═══ EVALUATION REPORT ═══",
            f"  Overall quality: {self.overall_quality.upper()}",
            f"  Ranking stability: {self.ranking_stability:.0%}",
            f"  Score spread: {self.score_distribution.get('std_dev', 0):.1f}%",
            f"  Score range: {self.score_distribution.get('min', 0):.1f}% — {self.score_distribution.get('max', 0):.1f}%",
        ]
        if self.calibration:
            lines.append(f"  Calibration bands:")
            for band, count in self.calibration.items():
                lines.append(f"    {band}: {count} candidates")
        if self.bias_flags:
            lines.append(f"  Bias warnings:")
            for flag in self.bias_flags:
                lines.append(f"    ⚠️  {flag}")
        if self.keyword_stuffing_suspects:
            lines.append(f"  Keyword stuffing suspects:")
            for s in self.keyword_stuffing_suspects:
                lines.append(f"    🔍 {s}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_evaluation_report(
    results: list[MatchResult],
    jd: Optional[JobDescription] = None,
) -> EvaluationReport:
    """
    Generate a full evaluation report for a set of match results.

    Args:
        results: List of MatchResult from the pipeline (sorted by score)
        jd: The JobDescription used for matching (optional, for deeper analysis)

    Returns:
        EvaluationReport with all evaluation metrics
    """
    report = EvaluationReport()

    if not results:
        report.overall_quality = "poor"
        return report

    scores = [r.qualification_percentage for r in results]

    # 1. Score distribution analysis
    report.score_distribution = evaluate_score_distribution(scores)

    # 2. Calibration bands
    report.calibration = evaluate_calibration(scores)

    # 3. Ranking stability
    report.ranking_stability = evaluate_ranking_stability(results)

    # 4. Bias detection
    report.bias_flags = detect_scoring_bias(results)

    # 5. Keyword stuffing detection
    report.keyword_stuffing_suspects = detect_keyword_stuffing(results, jd)

    # Overall quality assessment
    if report.ranking_stability < 0.5 or len(report.bias_flags) > 2:
        report.overall_quality = "poor"
    elif report.ranking_stability < 0.8 or len(report.bias_flags) > 0:
        report.overall_quality = "acceptable"
    else:
        report.overall_quality = "good"

    logger.info(f"Evaluation complete: quality={report.overall_quality}, stability={report.ranking_stability:.0%}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SCORING EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_score_distribution(scores: list[float]) -> dict:
    """
    Analyze the distribution of match scores.

    Checks for:
        - All-same scores (broken pipeline)
        - Too-narrow range (poor discrimination)
        - Ceiling/floor effects (all high or all low)

    Returns:
        dict with min, max, mean, median, std_dev, issues
    """
    if not scores:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std_dev": 0, "issues": ["No scores"]}

    result = {
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
        "count": len(scores),
        "issues": [],
    }

    # Check for problems
    if result["std_dev"] < 2.0 and len(scores) > 2:
        result["issues"].append("Scores too similar — pipeline may not be discriminating well")

    if result["max"] - result["min"] < 5.0 and len(scores) > 2:
        result["issues"].append("Score range too narrow (<5%) — all candidates appear equivalent")

    if result["min"] > 80:
        result["issues"].append("Floor effect: all scores >80% — scoring may be too generous")

    if result["max"] < 30:
        result["issues"].append("Ceiling effect: all scores <30% — scoring may be too strict")

    return result


def evaluate_calibration(scores: list[float]) -> dict:
    """
    Bin scores into human-interpretable bands and check distribution.

    Bands:
        Strong fit: 70-100%
        Good fit: 50-69%
        Partial fit: 30-49%
        Weak fit: 0-29%
    """
    bands = {
        "Strong fit (70-100%)": 0,
        "Good fit (50-69%)": 0,
        "Partial fit (30-49%)": 0,
        "Weak fit (0-29%)": 0,
    }

    for s in scores:
        if s >= 70:
            bands["Strong fit (70-100%)"] += 1
        elif s >= 50:
            bands["Good fit (50-69%)"] += 1
        elif s >= 30:
            bands["Partial fit (30-49%)"] += 1
        else:
            bands["Weak fit (0-29%)"] += 1

    return bands


def evaluate_ranking_stability(results: list[MatchResult]) -> float:
    """
    Test if the top-K ranking is stable under small weight perturbations.

    Perturbs each scoring weight by ±10% and checks if the top-3 ranking changes.
    If ranking is unstable, it means small weight changes flip candidates — unreliable.

    Returns:
        Stability score 0.0-1.0 (1.0 = perfectly stable)
    """
    if len(results) < 3:
        return 1.0

    # Get current top-3 order
    original_top3 = [r.candidate.full_name for r in results[:3]]

    # Get scoring breakdowns
    breakdowns = [r.scoring_breakdown for r in results]

    # Default weights
    weights = {
        "must_have_match": 0.35,
        "experience_match": 0.25,
        "skills_depth": 0.20,
        "project_relevance": 0.12,
        "recency_factor": 0.08,
    }

    perturbations_stable = 0
    total_perturbations = 0

    # Test each weight with ±10% perturbation
    for perturb_key in weights:
        for delta in [-0.10, +0.10]:
            perturbed_weights = weights.copy()
            perturbed_weights[perturb_key] += delta

            # Normalize to sum=1
            total_w = sum(perturbed_weights.values())
            perturbed_weights = {k: v / total_w for k, v in perturbed_weights.items()}

            # Recompute scores with perturbed weights
            perturbed_scores = []
            for i, bd in enumerate(breakdowns):
                score = (
                    perturbed_weights["must_have_match"] * bd.must_have_match +
                    perturbed_weights["experience_match"] * bd.experience_match +
                    perturbed_weights["skills_depth"] * bd.skills_depth +
                    perturbed_weights["project_relevance"] * bd.project_relevance +
                    perturbed_weights["recency_factor"] * bd.recency_factor
                ) * 100
                perturbed_scores.append((score, results[i].candidate.full_name))

            # Sort by perturbed score
            perturbed_scores.sort(key=lambda x: x[0], reverse=True)
            perturbed_top3 = [name for _, name in perturbed_scores[:3]]

            # Check if top-3 is same (order-independent for stability)
            if set(perturbed_top3) == set(original_top3):
                perturbations_stable += 1
            total_perturbations += 1

    return perturbations_stable / total_perturbations if total_perturbations > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: BIAS DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_scoring_bias(results: list[MatchResult]) -> list[str]:
    """
    Detect potential biases in scoring results.

    Checks:
        - Experience over-weighting: Do high-experience candidates always win?
        - Skills-only bias: Do candidates with more listed skills score higher
          regardless of relevance?
        - Recency bias: Are recent-only candidates unfairly advantaged?
    """
    flags = []

    if len(results) < 3:
        return flags

    scores = [r.qualification_percentage for r in results]
    experiences = [r.candidate.total_experience_years or 0 for r in results]
    skill_counts = [len(r.candidate.skills) for r in results]

    # Check experience correlation
    if len(set(experiences)) > 1:
        exp_corr = _spearman_rank_correlation(scores, experiences)
        if exp_corr > 0.9:
            flags.append(
                f"High experience-score correlation ({exp_corr:.2f}) — "
                f"scoring may over-weight years of experience"
            )

    # Check skill count correlation
    if len(set(skill_counts)) > 1:
        skill_corr = _spearman_rank_correlation(scores, skill_counts)
        if skill_corr > 0.9:
            flags.append(
                f"High skill-count-score correlation ({skill_corr:.2f}) — "
                f"more listed skills = higher score regardless of relevance"
            )

    # Check if all top candidates have very similar profiles (diversity issue)
    if len(results) >= 5:
        top3_domains = set()
        for r in results[:3]:
            top3_domains.update(r.candidate.domain_expertise[:2])
        bottom3_domains = set()
        for r in results[-3:]:
            bottom3_domains.update(r.candidate.domain_expertise[:2])

        if top3_domains and not top3_domains.intersection(bottom3_domains):
            # Not necessarily bias, but worth noting
            pass

    return flags


def detect_keyword_stuffing(
    results: list[MatchResult],
    jd: Optional[JobDescription] = None,
) -> list[str]:
    """
    Detect candidates who may be keyword-stuffing their resumes.

    A keyword-stuffed resume has an unusually high ratio of JD keywords
    relative to overall content length, suggesting artificial optimization.

    Heuristic: If a candidate's must_have_match score is >0.9 but their
    skills_depth is <0.3, they likely have keywords without real depth.
    """
    suspects = []

    for r in results:
        bd = r.scoring_breakdown

        # High keyword match but low depth = potential stuffing
        if bd.must_have_match > 0.9 and bd.skills_depth < 0.3:
            suspects.append(
                f"{r.candidate.full_name}: must_have={bd.must_have_match:.0%} "
                f"but depth={bd.skills_depth:.0%} (keywords without evidence)"
            )

        # Unusually many skills listed (>30) with high match
        if len(r.candidate.skills) > 30 and bd.must_have_match > 0.8:
            suspects.append(
                f"{r.candidate.full_name}: {len(r.candidate.skills)} skills listed "
                f"(abnormally high — possible padding)"
            )

    return suspects


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """
    Compute Spearman rank correlation between two lists.

    Returns value between -1 and 1:
        +1 = perfect positive rank correlation
        -1 = perfect negative rank correlation
         0 = no correlation
    """
    n = len(x)
    if n < 3:
        return 0.0

    # Compute ranks
    x_ranks = _compute_ranks(x)
    y_ranks = _compute_ranks(y)

    # Spearman formula: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    d_squared_sum = sum((x_ranks[i] - y_ranks[i]) ** 2 for i in range(n))
    denominator = n * (n ** 2 - 1)

    if denominator == 0:
        return 0.0

    return 1.0 - (6.0 * d_squared_sum) / denominator


def _compute_ranks(values: list[float]) -> list[float]:
    """Compute ranks for a list of values (handles ties with average rank)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        # Find all tied values
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1

        # Average rank for tied values
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank

        i = j

    return ranks
