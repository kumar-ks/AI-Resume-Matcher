"""
Stage 1: JD Understanding
==========================

PURPOSE:
    Extracts structured requirements from raw job description text using an LLM.
    Identifies must-have skills, good-to-have skills, experience range,
    education requirements, domain/industry, and certifications.

CALL HIERARCHY:
    Called by:
        - pipeline.py → calls JDUnderstanding.extract() as Stage 1 of the matching pipeline
    Calls internally:
        - litellm.acompletion() → sends JD text to LLM for structured extraction
        - _extract_json() → parses raw LLM response text into a Python dict
        - _try_parse_json() → attempts JSON parsing with fixups for common LLM issues
        - _parse_response() → converts parsed dict into a JobDescription model
        - _parse_role_level() → maps string role level to ExperienceLevel enum
        - _fallback_extraction() → provides minimal extraction when LLM fails

RETURNS:
    JobDescription model with all extracted fields populated (title, skills,
    experience range, education, domain, certifications, responsibilities, etc.)

FLOW:
    1. Receive raw JD text from pipeline.py
    2. Format the extraction prompt with the JD text
    3. Send prompt to LLM via litellm.acompletion()
    4. Parse LLM response JSON (with multiple fallback strategies)
    5. Map parsed data to JobDescription model
    6. Return structured JobDescription to pipeline.py
"""

import json
import logging
import re
from typing import Optional

import litellm

from matching_engine.models import (
    ExperienceLevel,
    JobDescription,
    Skill,
    SkillCategory,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
# This prompt instructs the LLM to act as an HR analyst and return structured
# JSON with all key fields extracted from the raw job description text.
# The double-braces {{ }} are used to escape Python's .format() method.
JD_EXTRACTION_PROMPT = """You are an expert HR analyst. Analyze the following job description and extract structured information.

Return a JSON object with these fields:
{{
    "title": "Job title",
    "must_have_skills": [{{"name": "skill name", "years_required": null or number}}],
    "good_to_have_skills": [{{"name": "skill name", "years_required": null or number}}],
    "experience_range_min": minimum years or null,
    "experience_range_max": maximum years or null,
    "education": ["degree requirements"],
    "domain_industry": ["relevant domains/industries"],
    "certifications": ["required or preferred certifications"],
    "responsibilities": ["key responsibilities"],
    "location": "location or null",
    "role_level": "entry|mid|senior|lead|principal"
}}

Job Description:
---
{jd_text}
---

Return ONLY valid JSON, no markdown formatting."""


class JDUnderstanding:
    """
    Parses raw job description text into a structured JobDescription model.

    Uses an LLM to extract key requirements including skills (must-have vs
    good-to-have), experience range, education, domain, and certifications.

    This is the entry point for Stage 1 of the matching pipeline.
    pipeline.py instantiates this class and calls extract() with raw JD text.
    """

    def __init__(self, model: str = "ollama/llama2", temperature: float = 0.1):
        """
        Initialize JD Understanding stage.

        Called by: pipeline.py during pipeline initialization.

        Args:
            model: LiteLLM model identifier (e.g., "ollama/llama2", "gpt-4")
            temperature: LLM temperature for extraction (lower = more deterministic)
        """
        self.model = model
        self.temperature = temperature
        logger.debug(f"JDUnderstanding initialized with model={model}, temperature={temperature}")

    async def extract(self, jd_text: str) -> JobDescription:
        """
        Extract structured requirements from raw JD text.

        Called by: pipeline.py as Stage 1 of the matching pipeline.
        Calls: litellm.acompletion(), _extract_json(), _parse_response(), _fallback_extraction()

        Flow:
            1. Validate input text is not empty
            2. Format the extraction prompt with JD text
            3. Call LLM via litellm.acompletion()
            4. Extract JSON from LLM response
            5. Parse JSON into JobDescription model
            6. On failure, fall back to keyword-based extraction

        Args:
            jd_text: Raw job description text (from PDF/DOCX/plain text)

        Returns:
            JobDescription with all extracted fields populated
        """
        logger.info("Stage 1: Extracting JD requirements")
        logger.debug(f"JD text length: {len(jd_text)} characters")

        # Guard clause: if JD text is empty/whitespace, return empty model
        if not jd_text.strip():
            logger.warning("Empty JD text provided")
            return JobDescription(raw_text=jd_text)

        try:
            # ── Step 1: Call LLM with the extraction prompt ──
            logger.debug(f"Sending JD extraction request to model: {self.model}")
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "user", "content": JD_EXTRACTION_PROMPT.format(jd_text=jd_text)}
                ],
                temperature=self.temperature,
                max_tokens=4096,
            )

            # ── Step 2: Extract raw content from LLM response ──
            content = response.choices[0].message.content
            logger.debug(f"LLM response received, content length: {len(content) if content else 0}")

            # ── Step 3: Parse JSON from the response content ──
            data = self._extract_json(content)
            if data is None:
                logger.error("Could not extract valid JSON from LLM response for JD")
                return self._fallback_extraction(jd_text)

            # ── Step 4: Convert parsed dict to JobDescription model ──
            logger.debug(f"Successfully parsed JD JSON with keys: {list(data.keys())}")
            return self._parse_response(data, jd_text)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_extraction(jd_text)
        except Exception as e:
            logger.error(f"JD extraction failed: {e}")
            return self._fallback_extraction(jd_text)

    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from LLM response, handling markdown and formatting.

        Called by: extract()
        Calls: _try_parse_json()

        Strategy (tried in order):
            1. Direct json.loads() on the full text
            2. Look for ```json ... ``` markdown code blocks
            3. Find the outermost { ... } brace pair via regex

        Args:
            text: Raw LLM response text

        Returns:
            Parsed dict if successful, None otherwise
        """
        if not text:
            logger.debug("_extract_json received empty text")
            return None

        # Strategy 1: Try direct JSON parse (ideal case - LLM returned pure JSON)
        try:
            logger.debug("Attempting direct JSON parse of LLM response")
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Direct JSON parse failed, trying markdown extraction")
            pass

        # Strategy 2: Look for JSON inside markdown code blocks (```json ... ```)
        json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_block:
            raw = json_block.group(1).strip()
            logger.debug(f"Found markdown JSON block, length: {len(raw)}")
            parsed = self._try_parse_json(raw)
            if parsed is not None:
                return parsed

        # Strategy 3: Find outermost braces and try to parse that substring
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            logger.debug("Attempting brace-match JSON extraction")
            parsed = self._try_parse_json(brace_match.group(0))
            if parsed is not None:
                return parsed

        logger.debug("All JSON extraction strategies failed")
        return None

    def _try_parse_json(self, raw: str) -> Optional[dict]:
        """
        Try to parse JSON with fixups for common LLM issues.

        Called by: _extract_json()

        Strategies:
            1. Direct parse
            2. Remove trailing commas before } or ] (common LLM mistake)
            3. Truncate from the end to find the last valid closing brace

        Args:
            raw: Raw JSON string to parse

        Returns:
            Parsed dict if successful, None otherwise
        """
        # Attempt 1: Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Fix trailing commas (e.g., {"a": 1,} → {"a": 1})
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            logger.debug("Trying JSON parse after removing trailing commas")
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Attempt 3: Truncate from end to find last valid JSON object
        # This handles cases where LLM appends extra text after the JSON
        for i in range(len(raw) - 1, 0, -1):
            if raw[i] == "}":
                try:
                    return json.loads(raw[: i + 1])
                except json.JSONDecodeError:
                    continue

        logger.debug("All JSON parse attempts failed for this raw string")
        return None

    def _parse_response(self, data: dict, raw_text: str) -> JobDescription:
        """
        Parse LLM JSON response into JobDescription model.

        Called by: extract()
        Calls: _parse_role_level()

        Maps the flat JSON dict from the LLM into the structured JobDescription
        Pydantic model, converting skill dicts into Skill objects with categories.

        Args:
            data: Parsed JSON dict from LLM response
            raw_text: Original raw JD text (preserved in the model)

        Returns:
            Fully populated JobDescription model
        """
        logger.debug(f"Parsing JD response: title='{data.get('title', '')}'")

        # ── Parse must-have skills into Skill objects ──
        # Each skill can be a dict {"name": ..., "years_required": ...} or a plain string
        must_have = [
            Skill(
                name=s.get("name", "") if isinstance(s, dict) else str(s),
                category=SkillCategory.MUST_HAVE,
                years_required=s.get("years_required") if isinstance(s, dict) else None,
            )
            for s in data.get("must_have_skills", [])
        ]
        logger.debug(f"Extracted {len(must_have)} must-have skills")

        # ── Parse good-to-have skills into Skill objects ──
        good_to_have = [
            Skill(
                name=s.get("name", "") if isinstance(s, dict) else str(s),
                category=SkillCategory.GOOD_TO_HAVE,
                years_required=s.get("years_required") if isinstance(s, dict) else None,
            )
            for s in data.get("good_to_have_skills", [])
        ]
        logger.debug(f"Extracted {len(good_to_have)} good-to-have skills")

        # ── Parse role level string to enum ──
        role_level = self._parse_role_level(data.get("role_level"))
        logger.debug(f"Parsed role level: {role_level}")

        # ── Assemble and return the JobDescription model ──
        return JobDescription(
            title=data.get("title", ""),
            must_have_skills=must_have,
            good_to_have_skills=good_to_have,
            experience_range_min=data.get("experience_range_min"),
            experience_range_max=data.get("experience_range_max"),
            education=data.get("education", []),
            domain_industry=data.get("domain_industry", []),
            certifications=data.get("certifications", []),
            responsibilities=data.get("responsibilities", []),
            location=data.get("location"),
            role_level=role_level,
            raw_text=raw_text,
        )

    def _parse_role_level(self, level: Optional[str]) -> Optional[ExperienceLevel]:
        """
        Map string role level to ExperienceLevel enum.

        Called by: _parse_response()

        Args:
            level: String like "entry", "mid", "senior", "lead", "principal"

        Returns:
            Corresponding ExperienceLevel enum value, or None if not recognized
        """
        if not level:
            logger.debug("No role level provided, returning None")
            return None

        mapping = {
            "entry": ExperienceLevel.ENTRY,
            "mid": ExperienceLevel.MID,
            "senior": ExperienceLevel.SENIOR,
            "lead": ExperienceLevel.LEAD,
            "principal": ExperienceLevel.PRINCIPAL,
        }
        result = mapping.get(level.lower())
        logger.debug(f"Role level '{level}' mapped to: {result}")
        return result

    def _fallback_extraction(self, jd_text: str) -> JobDescription:
        """
        Basic keyword-based extraction when LLM fails.

        Called by: extract() when LLM call or JSON parsing fails.

        Provides a minimal extraction using simple heuristics.
        The raw_text is preserved so downstream stages can still
        use the full text for semantic matching.

        Args:
            jd_text: Original raw JD text

        Returns:
            Minimal JobDescription with only raw_text populated
        """
        logger.info("Using fallback keyword extraction for JD")
        logger.debug(f"Fallback triggered, preserving raw text of {len(jd_text)} chars")
        return JobDescription(raw_text=jd_text)
