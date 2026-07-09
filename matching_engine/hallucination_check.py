"""
Hallucination Check — Grounding Verification
==============================================

Verifies that LLM-extracted fields actually exist in the source resume text.
Detects when the LLM invents skills, companies, certifications, or experience
that don't appear in the original document.

WHAT IT CHECKS:
    - Skills: Does each extracted skill appear (fuzzy) in the resume text?
    - Companies: Does each company name appear in the resume text?
    - Certifications: Does each certification appear in the resume text?
    - Experience years: Is the claimed total plausible given work history dates?
    - Name: Does the extracted name appear in the resume text?

SCORING:
    Each field gets a grounding_score (0.0 to 1.0):
        1.0 = fully grounded (all items found in source text)
        0.0 = fully hallucinated (no items found in source text)

    Overall confidence = weighted average of field scores.

OUTPUT:
    HallucinationReport with per-field scores, flagged items, and overall confidence.

CALLED BY:
    - scanner.py → after LLM extraction, before storing in DB
    - Can also be used standalone for auditing existing profiles

PHILOSOPHY:
    - Better to flag a false positive than miss a hallucination
    - Fuzzy matching accounts for OCR errors, abbreviations, case differences
    - Does NOT block storage — just flags and logs warnings
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FieldGrounding:
    """Grounding result for a single field."""
    field_name: str
    total_items: int = 0
    grounded_items: int = 0
    hallucinated_items: list[str] = field(default_factory=list)
    score: float = 1.0  # 1.0 = fully grounded


@dataclass
class HallucinationReport:
    """Full hallucination check report for a profile extraction."""
    overall_confidence: float = 1.0  # 0.0 to 1.0
    fields: dict[str, FieldGrounding] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    is_reliable: bool = True  # False if confidence < threshold

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Grounding confidence: {self.overall_confidence:.0%}"]
        for name, fg in self.fields.items():
            if fg.hallucinated_items:
                lines.append(f"  {name}: {fg.score:.0%} grounded ({len(fg.hallucinated_items)} suspect)")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def check_hallucination(profile, raw_text: str, confidence_threshold: float = 0.6) -> HallucinationReport:
    """
    Verify that LLM-extracted profile fields are grounded in the source text.

    Args:
        profile: ResumeProfile extracted by Stage 2
        raw_text: Original resume text (before LLM processing)
        confidence_threshold: Below this, profile is flagged as unreliable

    Returns:
        HallucinationReport with per-field scores and overall confidence
    """
    if not raw_text or not raw_text.strip():
        return HallucinationReport(
            overall_confidence=0.0,
            is_reliable=False,
            warnings=["No source text available for grounding check"],
        )

    # Normalize source text for matching
    source_lower = raw_text.lower()
    source_normalized = _normalize_text(raw_text)

    report = HallucinationReport()

    # Check each field
    report.fields["skills"] = _check_skills(profile.skills, source_lower, source_normalized)
    report.fields["companies"] = _check_companies(profile.work_experiences, source_lower)
    report.fields["certifications"] = _check_certifications(profile.certifications, source_lower)
    report.fields["experience_years"] = _check_experience_years(profile, source_lower)
    report.fields["name"] = _check_name(profile, source_lower)

    # Compute overall confidence (weighted average)
    weights = {
        "skills": 0.30,
        "companies": 0.30,
        "certifications": 0.15,
        "experience_years": 0.15,
        "name": 0.10,
    }

    total_weight = 0.0
    weighted_score = 0.0
    for field_name, fg in report.fields.items():
        w = weights.get(field_name, 0.1)
        weighted_score += fg.score * w
        total_weight += w

    report.overall_confidence = weighted_score / total_weight if total_weight > 0 else 0.0
    report.is_reliable = report.overall_confidence >= confidence_threshold

    # Generate warnings for badly grounded fields
    for field_name, fg in report.fields.items():
        if fg.score < 0.5 and fg.total_items > 0:
            report.warnings.append(
                f"{field_name}: only {fg.grounded_items}/{fg.total_items} items found in source text "
                f"(suspect: {', '.join(fg.hallucinated_items[:5])})"
            )

    if not report.is_reliable:
        logger.warning(
            f"Hallucination check FAILED (confidence={report.overall_confidence:.0%}): "
            f"{'; '.join(report.warnings)}"
        )
    else:
        logger.debug(f"Hallucination check passed (confidence={report.overall_confidence:.0%})")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# FIELD-LEVEL CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _check_skills(skills: list[str], source_lower: str, source_normalized: str) -> FieldGrounding:
    """
    Check if extracted skills appear in the resume text.

    Handles compound skill strings like "Microservices Development: Java - Spring - Docker"
    by splitting them into individual terms and checking each sub-term.
    A compound skill is considered grounded if >50% of its sub-terms are found.
    """
    if not skills:
        return FieldGrounding(field_name="skills", score=1.0)

    grounded = 0
    hallucinated = []

    for skill in skills:
        if _is_skill_grounded(skill, source_lower, source_normalized):
            grounded += 1
        else:
            hallucinated.append(skill)

    score = grounded / len(skills) if skills else 1.0

    return FieldGrounding(
        field_name="skills",
        total_items=len(skills),
        grounded_items=grounded,
        hallucinated_items=hallucinated,
        score=score,
    )


def _is_skill_grounded(skill: str, source_lower: str, source_normalized: str) -> bool:
    """
    Check if a skill (possibly compound) is grounded in source text.

    For compound skills like "Java - Spring - Docker - Kubernetes":
        Split into sub-terms and check if >50% are found individually.

    For simple skills like "Python":
        Direct fuzzy match.
    """
    # First try direct match
    if _fuzzy_find(skill, source_lower, source_normalized):
        return True

    # Split compound skills on common delimiters: - , : / |
    import re
    sub_terms = re.split(r'[\-,:/|•·]', skill)
    sub_terms = [t.strip() for t in sub_terms if t.strip() and len(t.strip()) >= 2]

    if len(sub_terms) <= 1:
        # Not compound, and direct match already failed
        return False

    # For compound skills, check if >50% of sub-terms are found
    found_count = sum(1 for t in sub_terms if _fuzzy_find(t, source_lower, source_normalized))
    return found_count / len(sub_terms) >= 0.5

    score = grounded / len(skills) if skills else 1.0

    return FieldGrounding(
        field_name="skills",
        total_items=len(skills),
        grounded_items=grounded,
        hallucinated_items=hallucinated,
        score=score,
    )


def _check_companies(work_experiences: list, source_lower: str) -> FieldGrounding:
    """Check if extracted company names appear in the resume text."""
    companies = [exp.company for exp in work_experiences if exp.company]

    if not companies:
        return FieldGrounding(field_name="companies", score=1.0)

    grounded = 0
    hallucinated = []

    for company in companies:
        # Companies may appear abbreviated or with different casing
        if _fuzzy_find(company, source_lower, source_lower):
            grounded += 1
        else:
            hallucinated.append(company)

    score = grounded / len(companies) if companies else 1.0

    return FieldGrounding(
        field_name="companies",
        total_items=len(companies),
        grounded_items=grounded,
        hallucinated_items=hallucinated,
        score=score,
    )


def _check_certifications(certifications: list[str], source_lower: str) -> FieldGrounding:
    """Check if extracted certifications appear in the resume text."""
    if not certifications:
        return FieldGrounding(field_name="certifications", score=1.0)

    grounded = 0
    hallucinated = []

    for cert in certifications:
        if _fuzzy_find(cert, source_lower, source_lower):
            grounded += 1
        else:
            hallucinated.append(cert)

    score = grounded / len(certifications) if certifications else 1.0

    return FieldGrounding(
        field_name="certifications",
        total_items=len(certifications),
        grounded_items=grounded,
        hallucinated_items=hallucinated,
        score=score,
    )


def _check_experience_years(profile, source_lower: str) -> FieldGrounding:
    """
    Check if claimed experience years is plausible.

    Heuristics:
        - If work experiences have dates, compute actual span
        - If claimed years differ by >5 from computed span, flag it
        - If no dates available, check if a number close to claimed years appears in text
    """
    claimed = profile.total_experience_years
    if claimed is None:
        return FieldGrounding(field_name="experience_years", score=1.0)

    # Try to compute from work history dates
    if profile.work_experiences:
        years_with_dates = [exp for exp in profile.work_experiences if exp.start_year]
        if years_with_dates:
            earliest = min(exp.start_year for exp in years_with_dates)
            latest_end = max(
                (exp.end_year or 2024) for exp in years_with_dates
            )
            computed_span = latest_end - earliest

            deviation = abs(claimed - computed_span)
            if deviation <= 3:
                score = 1.0
            elif deviation <= 5:
                score = 0.7
            else:
                score = 0.3

            fg = FieldGrounding(
                field_name="experience_years",
                total_items=1,
                grounded_items=1 if score >= 0.7 else 0,
                score=score,
            )
            if score < 0.7:
                fg.hallucinated_items.append(
                    f"Claimed {claimed}yrs but work history spans ~{computed_span}yrs"
                )
            return fg

    # Fallback: check if the number appears somewhere in text
    claimed_str = str(int(claimed))
    if claimed_str in source_lower or f"{claimed}" in source_lower:
        return FieldGrounding(field_name="experience_years", total_items=1, grounded_items=1, score=1.0)

    # Can't verify — give benefit of doubt
    return FieldGrounding(field_name="experience_years", total_items=1, grounded_items=1, score=0.8)


def _check_name(profile, source_lower: str) -> FieldGrounding:
    """Check if extracted name appears in the resume text."""
    name_parts = [profile.first_name, profile.last_name]
    name_parts = [p for p in name_parts if p and p.strip()]

    if not name_parts:
        return FieldGrounding(field_name="name", score=0.5, hallucinated_items=["No name extracted"])

    grounded = sum(1 for part in name_parts if part.lower() in source_lower)
    score = grounded / len(name_parts)

    fg = FieldGrounding(
        field_name="name",
        total_items=len(name_parts),
        grounded_items=grounded,
        score=score,
    )
    if score < 1.0:
        missing = [p for p in name_parts if p.lower() not in source_lower]
        fg.hallucinated_items = missing

    return fg


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s+#.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _fuzzy_find(term: str, source_lower: str, source_normalized: str) -> bool:
    """
    Check if a term appears in source text using fuzzy matching.

    Strategies:
        1. Exact substring match (case-insensitive)
        2. Normalized match (strip punctuation)
        3. Token-level match (all significant words of the term appear)
        4. Abbreviation match (e.g., "AWS" matches "Amazon Web Services")
    """
    term_lower = term.lower().strip()

    if not term_lower or len(term_lower) < 2:
        return True  # Skip trivially short terms

    # Strategy 1: Exact substring
    if term_lower in source_lower:
        return True

    # Strategy 2: Normalized
    term_normalized = re.sub(r'[^a-z0-9\s+#.]', ' ', term_lower).strip()
    if term_normalized and term_normalized in source_normalized:
        return True

    # Strategy 3: Token-level (all significant words present)
    tokens = [t for t in term_lower.split() if len(t) >= 3]
    if tokens and len(tokens) <= 4:
        # All tokens must be present in source
        if all(t in source_lower for t in tokens):
            return True

    # Strategy 4: Abbreviation / acronym (e.g., "CI/CD" → "ci" and "cd")
    if "/" in term_lower:
        parts = [p.strip() for p in term_lower.split("/") if p.strip()]
        if all(p in source_lower for p in parts):
            return True

    # Strategy 5: Common variations (e.g., "kubernetes" ↔ "k8s")
    aliases = _get_common_aliases(term_lower)
    for alias in aliases:
        if alias in source_lower:
            return True

    return False


def _get_common_aliases(term: str) -> list[str]:
    """Return common aliases/abbreviations for tech terms."""
    _ALIASES = {
        "kubernetes": ["k8s"],
        "k8s": ["kubernetes"],
        "javascript": ["js"],
        "typescript": ["ts"],
        "python": ["py"],
        "machine learning": ["ml"],
        "artificial intelligence": ["ai"],
        "natural language processing": ["nlp"],
        "continuous integration": ["ci"],
        "continuous deployment": ["cd"],
        "ci/cd": ["cicd", "ci cd", "continuous integration"],
        "amazon web services": ["aws"],
        "aws": ["amazon web services"],
        "google cloud platform": ["gcp"],
        "gcp": ["google cloud"],
        "microsoft azure": ["azure"],
        "devops": ["dev ops"],
        "devsecops": ["dev sec ops", "devsecops"],
        "docker": ["containerization"],
        "terraform": ["iac", "infrastructure as code"],
        "react": ["reactjs", "react.js"],
        "node": ["nodejs", "node.js"],
        "postgres": ["postgresql"],
        "postgresql": ["postgres"],
        "mongodb": ["mongo"],
        "elasticsearch": ["elastic"],
    }
    return _ALIASES.get(term, [])
