"""
Shared Utilities for the AI Resume Matcher
============================================

This module consolidates duplicated utility logic that was previously repeated
across jd_understanding.py, resume_understanding.py, and explainability.py.

Functions provided:
    - extract_json_from_llm_response: Robust JSON extraction from LLM output
    - extract_name_from_text: Regex-based name detection from resume text
    - extract_email_from_text: Email address extraction via regex
    - extract_phone_from_text: Phone number extraction via regex
    - estimate_experience_from_text: Experience years estimation from text patterns

Called by:
    - matching_engine.jd_understanding (JDUnderstanding._extract_json)
    - matching_engine.resume_understanding (ResumeUnderstanding._extract_json,
      _extract_baseline)
    - matching_engine.explainability (ExplainabilityEngine._extract_json)
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# JSON EXTRACTION
# =============================================================================


def extract_json_from_llm_response(text: str) -> Optional[dict]:
    """
    Extract a JSON object from an LLM response string.

    LLMs are instructed to return pure JSON, but in practice they may:
      - Wrap it in ```json ... ``` or ``` ... ``` markdown blocks
      - Include preamble/postamble text around the JSON
      - Truncate the response mid-JSON (token limit)
      - Add trailing commas (invalid JSON)

    This function tries multiple strategies in order of likelihood:
      1. Direct json.loads() on the full text
      2. Extract content from markdown code blocks
      3. Find the first { ... } brace block via regex
      For each candidate string:
        a. Direct parse
        b. Remove trailing commas and parse
        c. Truncation recovery (walk backwards to find valid closing brace)

    Args:
        text: Raw LLM response string.

    Returns:
        Parsed dict if successful, None otherwise.
    """
    if not text:
        logger.debug("extract_json_from_llm_response: received empty text")
        return None

    # Strategy 1: Direct parse — the ideal case (LLM returned clean JSON)
    try:
        result = json.loads(text)
        logger.debug("extract_json_from_llm_response: direct json.loads() succeeded")
        return result
    except json.JSONDecodeError:
        logger.debug("extract_json_from_llm_response: direct parse failed, trying alternatives")

    # Strategy 2: Extract from markdown code block ```json ... ``` or ``` ... ```
    json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_block:
        raw = json_block.group(1).strip()
        logger.debug(
            "extract_json_from_llm_response: found markdown code block, length=%d",
            len(raw),
        )
        parsed = _try_parse_json(raw)
        if parsed is not None:
            logger.debug("extract_json_from_llm_response: markdown block parse succeeded")
            return parsed
        logger.debug("extract_json_from_llm_response: markdown block parse failed")

    # Strategy 3: Find the first { ... } block (greedy, handles preamble text)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        raw = brace_match.group(0)
        logger.debug(
            "extract_json_from_llm_response: found brace block, length=%d",
            len(raw),
        )
        parsed = _try_parse_json(raw)
        if parsed is not None:
            logger.debug("extract_json_from_llm_response: brace block parse succeeded")
            return parsed
        logger.debug("extract_json_from_llm_response: brace block parse failed")

    logger.debug("extract_json_from_llm_response: all strategies exhausted, returning None")
    return None


def _try_parse_json(raw: str) -> Optional[dict]:
    """
    Attempt to parse a JSON string with progressive fixups.

    Strategies (in order):
      1. Direct json.loads()
      2. Remove trailing commas before } or ] (common LLM mistake)
      3. Truncation recovery: walk backwards through the string, try parsing
         at each } position (handles responses cut off by token limits)

    Args:
        raw: A string that should contain JSON (possibly malformed).

    Returns:
        Parsed dict if any strategy works, None otherwise.
    """
    # Attempt 1: Direct parse
    try:
        result = json.loads(raw)
        logger.debug("_try_parse_json: direct parse succeeded")
        return result
    except json.JSONDecodeError:
        pass

    # Attempt 2: Remove trailing commas before } or ]
    # e.g., {"a": 1, "b": 2,} → {"a": 1, "b": 2}
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        result = json.loads(fixed)
        logger.debug("_try_parse_json: trailing-comma fix succeeded")
        return result
    except json.JSONDecodeError:
        pass

    # Attempt 3: Truncation recovery
    # Walk backwards, try parsing at each } position.
    # Handles cases where the LLM response was cut off mid-JSON.
    logger.debug(
        "_try_parse_json: attempting truncation recovery (string length=%d)",
        len(raw),
    )
    for i in range(len(raw) - 1, 0, -1):
        if raw[i] == "}":
            try:
                result = json.loads(raw[: i + 1])
                logger.debug(
                    "_try_parse_json: truncation recovery succeeded at position %d", i
                )
                return result
            except json.JSONDecodeError:
                continue

    logger.debug("_try_parse_json: all parse attempts failed")
    return None


# =============================================================================
# NAME EXTRACTION
# =============================================================================


def extract_name_from_text(text: str) -> str:
    """
    Extract a candidate's name from resume text using heuristic line scanning.

    Strategy:
      1. Scan the first 15 lines for a short line (2-4 title-case words)
         that isn't a section header, email, phone, URL, or date-containing line.
      2. If not found, broaden the scan to the first 30 lines with slightly
         relaxed constraints.

    Args:
        text: Raw resume text (from PDF/DOCX extraction).

    Returns:
        Extracted name string, or "" if no suitable line is found.
    """
    lines = text.strip().split("\n")

    # Common resume section headers (normalized: no spaces, lowercase)
    section_headers = {
        "summary", "profile", "resume", "curriculum", "objective",
        "career", "about", "contact", "experience", "education",
        "skills", "certifications", "projects", "professional",
        "references", "achievements", "highlights", "overview",
        "keyskills", "technicalskills", "coreskills", "softskills",
        "workexperience", "professionalexperience", "employmenthistory",
        "careersummary", "careerobjective", "personaldetails",
        "personalinfo", "declaration", "hobbies", "interests",
        "languages", "awards", "publications", "training",
    }

    # ── Pass 1: Scan first 15 lines for a clean name line ──
    for idx, line in enumerate(lines[:15]):
        line = line.strip()
        if not line:
            continue

        # Skip section headers (normalize spaces for OCR artifacts)
        normalized = re.sub(r"\s+", "", line).lower()
        if normalized in section_headers or any(
            normalized.startswith(h) for h in section_headers
        ):
            continue

        # Skip lines with email, phone, or URLs
        if "@" in line or re.search(r"\d{5,}", line) or "http" in line.lower():
            continue

        # Skip lines that are too long (paragraph text)
        if len(line) > 35:
            continue

        # Skip lines with dates, special chars
        if re.search(r"['\"\d]{2,}|[-–—].*\d{2}", line):
            continue

        # Skip lines ending with dash/colon (job titles, headers)
        if line.endswith("-") or line.endswith(":") or line.endswith("–"):
            continue

        # A name: 2-4 words, each starts with uppercase, min 2 chars each
        words = line.split()
        if 2 <= len(words) <= 4 and all(
            re.match(r"^[A-Z][a-zA-Z.\-]*$", w) and len(w) >= 2 for w in words
        ):
            logger.debug("extract_name_from_text: found at line %d: %r", idx, line)
            return line

    # ── Pass 2: Broader scan (first 30 lines) for capitalized name pattern ──
    for idx, line in enumerate(lines[:30]):
        line = line.strip()
        if not line or len(line) > 30 or len(line) < 5:
            continue

        # Skip section headers
        normalized = re.sub(r"\s+", "", line).lower()
        if normalized in section_headers or any(
            normalized.startswith(h) for h in section_headers
        ):
            continue

        # Skip lines with digits, email, special chars
        if re.search(r"[\d@#$%&*(){}[\]]", line):
            continue

        # 2-4 words where each starts with uppercase
        words = line.split()
        if 2 <= len(words) <= 4 and all(
            re.match(r"^[A-Z][a-zA-Z.\-]*$", w) for w in words
        ):
            logger.debug("extract_name_from_text: found (pass 2) at line %d: %r", idx, line)
            return line

    logger.debug("extract_name_from_text: no name found")
    return ""


# =============================================================================
# EMAIL EXTRACTION
# =============================================================================


def extract_email_from_text(text: str) -> Optional[str]:
    """
    Extract the first email address from text using regex.

    Pattern matches standard email format: user@domain.tld

    Args:
        text: Raw text to scan for email addresses.

    Returns:
        The first email address found, or None.
    """
    if not text:
        return None

    match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
    )
    if match:
        email = match.group(0)
        logger.debug("extract_email_from_text: found %r", email)
        return email

    logger.debug("extract_email_from_text: no email found")
    return None


# =============================================================================
# PHONE EXTRACTION
# =============================================================================


def extract_phone_from_text(text: str) -> Optional[str]:
    """
    Extract the first phone number from text using regex.

    Covers formats such as:
      - +91 98765 43210
      - 123-4567-8901
      - +1 555 123 4567

    Args:
        text: Raw text to scan for phone numbers.

    Returns:
        The first phone number found (stripped), or None.
    """
    if not text:
        return None

    match = re.search(
        r"(\+?\d{1,3}[\s\-]?\d{4,5}[\s\-]?\d{4,6})", text
    )
    if match:
        phone = match.group(0).strip()
        logger.debug("extract_phone_from_text: found %r", phone)
        return phone

    logger.debug("extract_phone_from_text: no phone found")
    return None


# =============================================================================
# EXPERIENCE ESTIMATION
# =============================================================================


def estimate_experience_from_text(text: str) -> Optional[float]:
    """
    Estimate total years of experience from raw text using regex patterns.

    Handles common resume phrasings like:
      - "10+ years of experience"
      - "over 12 years in IT"
      - "approximately 8 yrs experience"
      - "experience of 15+ years"
      - "decade of experience"

    Applies a sanity check: only values between 1 and 50 years are accepted.

    Args:
        text: Raw resume or profile text to scan.

    Returns:
        Estimated experience in years as float, or None if no pattern matched.
    """
    if not text:
        return None

    logger.debug(
        "estimate_experience_from_text: scanning %d chars of text", len(text)
    )

    # Each pattern targets a different phrasing style commonly found in resumes
    patterns = [
        # "10+ years of experience" / "10 years experience"
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
        # "over 12 years" / "more than 8 years"
        r"(?:over|more than|approximately|around|nearly)\s*(\d+)\s*(?:years?|yrs?)",
        # "10+ years in IT" / "8 years in software"
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:in\s+(?:IT|software|engineering|industry))",
        # "experience of 15+ years"
        r"experience\s*(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)",
        # "10+ years of professional experience"
        r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:professional|total|overall|hands-on)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            years = float(match.group(1))
            if 1 <= years <= 50:
                logger.debug(
                    "estimate_experience_from_text: pattern %r matched, years=%.1f",
                    pattern, years,
                )
                return years
            else:
                logger.debug(
                    "estimate_experience_from_text: pattern %r matched but value %.1f "
                    "outside 1-50 range",
                    pattern, years,
                )

    # Handle "decade" / "plus decade" as a special case (= 10 years)
    if re.search(r"(?:over|plus)?\s*decade\s*(?:of\s+)?experience", text, re.IGNORECASE):
        logger.debug("estimate_experience_from_text: 'decade' pattern matched, returning 10.0")
        return 10.0

    logger.debug("estimate_experience_from_text: no patterns matched")
    return None
