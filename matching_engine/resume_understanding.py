"""
Stage 2: Resume Understanding

Extracts structured candidate information from raw resume text using a
combination of regex-based baseline extraction and LLM-powered deep parsing.

=== CALL HIERARCHY & FLOW ===

    extract() [PUBLIC - main entry point, called by pipeline.py]
        │
        ├── _extract_baseline(text)
        │       └── _extract_name_from_text(text)
        │       └── _estimate_experience_from_text(text)
        │
        ├── _extract_via_llm(text)  [async, calls LLM with retries]
        │       └── _extract_json(content)
        │               └── _try_parse_json(raw)
        │       └── _parse_response(data, raw_text)
        │               └── _estimate_experience(work_experiences)
        │               └── _estimate_experience_from_text(raw_text)
        │
        └── _merge_profiles(baseline, llm_profile, raw_text)

=== DATA FLOW ===

1. extract() receives raw resume text (string from PDF/DOCX/text).
2. _extract_baseline() uses regex to pull name, email, phone, experience
   — this ALWAYS succeeds and provides a safety net.
3. _extract_via_llm() sends the text to an LLM, parses the JSON response
   into a full ResumeProfile — this MAY fail (network, bad JSON, etc.).
4. _merge_profiles() combines both: LLM data takes priority for rich fields
   (skills, work history, projects), baseline fills contact-info gaps.

=== OUTPUT ===

Returns a ResumeProfile (Pydantic model) with:
  - Contact info (name, email, phone, location)
  - Career summary, skills, certifications
  - Work experiences (company, title, duration, technologies)
  - Projects, education, domain expertise
  - total_experience_years (calculated or extracted)
  - raw_text (original input preserved)
"""

import json
import logging
import re
from typing import Optional

import litellm

from matching_engine.models import (
    Project,
    ResumeProfile,
    WorkExperience,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------------------------------
# This prompt is sent to the LLM in _extract_via_llm(). It instructs the model
# to return a JSON object with all structured resume fields. The double-braces
# {{ }} are Python f-string escapes so that the only interpolation is
# {resume_text} at the bottom.
# ---------------------------------------------------------------------------
RESUME_EXTRACTION_PROMPT = """You are an expert resume parser. Analyze the following resume and extract structured information.

Return a JSON object with these fields:
{{
    "first_name": "string",
    "middle_name": "string or null",
    "last_name": "string",
    "email": "email or null",
    "phone": "phone number or null",
    "location": "location or null",
    "career_summary": "brief career summary in 2-3 sentences",
    "skills": ["list of all technical and soft skills"],
    "total_experience_years": number or null,
    "work_experiences": [
        {{
            "company": "company name",
            "title": "job title",
            "duration_months": number or null,
            "start_year": number or null,
            "end_year": number or null,
            "is_current": boolean,
            "technologies": ["technologies used"],
            "responsibilities": ["key responsibilities"],
            "domain": "industry domain or null"
        }}
    ],
    "projects": [
        {{
            "name": "project name",
            "description": "brief description",
            "technologies": ["technologies used"],
            "duration_months": number or null,
            "role": "role in project or null"
        }}
    ],
    "education": ["degree - institution - year"],
    "certifications": ["certification names"],
    "domain_expertise": ["domains/industries the candidate has worked in"]
}}

Resume:
---
{resume_text}
---

Return ONLY valid JSON, no markdown formatting."""


class ResumeUnderstanding:
    """
    Parses raw resume text into a structured ResumeProfile model.

    Uses an LLM to extract career summary, skills, work experience,
    projects, education, certifications, and domain expertise.

    Two-pass strategy:
      Pass 1 (baseline): Regex extraction — fast, deterministic, always works.
      Pass 2 (LLM):      Deep extraction — richer data, may fail.
      Merge:             LLM wins for rich fields; baseline fills gaps.
    """

    def __init__(self, model: str = "ollama/llama2", temperature: float = 0.1):
        """
        Initialize Resume Understanding stage.

        Args:
            model: LiteLLM model identifier (e.g. "ollama/llama2",
                   "openai/gpt-4", "anthropic/claude-3")
            temperature: LLM temperature for extraction (low = deterministic)
        """
        self.model = model
        self.temperature = temperature
        logger.debug(
            "ResumeUnderstanding initialized: model=%s, temperature=%s",
            self.model, self.temperature,
        )

    # =========================================================================
    # PUBLIC METHOD — called by pipeline.py
    # =========================================================================
    async def extract(self, resume_text: str) -> ResumeProfile:
        """
        Extract structured profile from raw resume text.

        This is the MAIN ENTRY POINT for Stage 2. It orchestrates:
          1. _extract_baseline()  → regex-based contact info & experience
          2. _extract_via_llm()   → LLM-based full extraction (may fail)
          3. _merge_profiles()    → combine both, LLM priority

        Called by: matching_engine.pipeline (the orchestration layer)
        Calls:     _extract_baseline, _extract_via_llm, _merge_profiles

        Args:
            resume_text: Raw resume text (from PDF/DOCX/plain text)

        Returns:
            ResumeProfile with all extracted fields populated
        """
        logger.info("Stage 2: Extracting resume information")
        logger.debug(
            "extract() called with text length=%d chars", len(resume_text)
        )

        # Guard: empty input produces an empty profile immediately
        if not resume_text.strip():
            logger.warning("Empty resume text provided")
            return ResumeProfile(raw_text=resume_text)

        # Step 1: Regex-based extraction (always succeeds, provides baseline)
        baseline = self._extract_baseline(resume_text)
        logger.debug(
            "Baseline extraction result: name=%r, email=%r, phone=%r, exp=%s",
            baseline["name"], baseline["email"],
            baseline["phone"], baseline["total_experience_years"],
        )

        # Step 2: LLM-based extraction (may fail, provides richer data)
        llm_profile = await self._extract_via_llm(resume_text)
        logger.debug(
            "LLM extraction result: %s",
            f"full_name={llm_profile.full_name!r}, skills_count={len(llm_profile.skills)}, "
            f"work_exp_count={len(llm_profile.work_experiences)}"
            if llm_profile else "None (LLM failed)",
        )

        # Step 3: Merge — LLM data takes priority, baseline fills gaps
        merged = self._merge_profiles(baseline, llm_profile, resume_text)
        logger.debug(
            "Merged profile: full_name=%r, email=%r, skills=%d, experiences=%d",
            merged.full_name, merged.email,
            len(merged.skills), len(merged.work_experiences),
        )
        return merged

    # =========================================================================
    # BASELINE EXTRACTION — regex-based, deterministic
    # =========================================================================
    def _extract_baseline(self, text: str) -> dict:
        """
        Extract basic contact info and experience using regex patterns.
        This always produces results regardless of LLM availability.

        Called by: extract()
        Calls:     _extract_name_from_text(), _estimate_experience_from_text()

        Returns:
            dict with keys: name, email, phone, total_experience_years
        """
        logger.debug("_extract_baseline(): starting regex-based extraction")

        # --- Extract email ---
        # Pattern: standard email format (user@domain.tld)
        email_match = re.search(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
        )
        email = email_match.group(0) if email_match else None
        logger.debug("Baseline email regex matched: %r", email)

        # --- Extract phone ---
        # Pattern: optional country code, groups of 4-6 digits separated by
        # spaces or hyphens. Covers formats like +91 98765 43210, 123-4567-8901
        phone_match = re.search(
            r"(\+?\d{1,3}[\s\-]?\d{4,5}[\s\-]?\d{4,6})", text
        )
        phone = phone_match.group(0).strip() if phone_match else None
        logger.debug("Baseline phone regex matched: %r", phone)

        # --- Extract name ---
        # Delegates to _extract_name_from_text which scans the first few lines
        name = self._extract_name_from_text(text)
        logger.debug("Baseline name extracted: %r", name)

        # --- Extract experience ---
        # Delegates to _estimate_experience_from_text which uses multiple
        # regex patterns to find phrases like "10+ years of experience"
        experience = self._estimate_experience_from_text(text)
        logger.debug("Baseline experience estimated: %s years", experience)

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "total_experience_years": experience,
        }

    def _extract_name_from_text(self, text: str) -> str:
        """
        Extract candidate name from the beginning of resume text.

        Heuristic: The name is typically one of the first non-empty lines
        that is short (1-5 words), contains only letters/dots/hyphens,
        and doesn't look like a section header or contain contact info.

        Called by: _extract_baseline()
        Calls:     nothing (leaf method)

        Returns:
            Extracted name string, or "" if no suitable line found
        """
        lines = text.strip().split("\n")
        logger.debug(
            "_extract_name_from_text(): scanning first 5 lines of %d total",
            len(lines),
        )

        for idx, line in enumerate(lines[:5]):  # Name is usually in first 5 lines
            line = line.strip()
            if not line:
                continue

            # Skip lines that look like section headers/labels
            if re.match(
                r"^(summary|profile|resume|curriculum|objective|career|about|contact|experience|education)",
                line,
                re.IGNORECASE,
            ):
                logger.debug("  Line %d skipped (section header): %r", idx, line)
                continue

            # Skip lines containing email addresses or long digit sequences (phone)
            if "@" in line or re.search(r"\d{5,}", line):
                logger.debug("  Line %d skipped (contains email/phone): %r", idx, line)
                continue

            # A name line is typically short (2-5 words), mostly letters
            words = line.split()
            if 1 <= len(words) <= 5 and all(
                re.match(r"^[A-Za-z.\-]+$", w) for w in words
            ):
                logger.debug("  Line %d accepted as name: %r", idx, line)
                return line

            logger.debug("  Line %d rejected (word count/chars): %r", idx, line)

        logger.debug("_extract_name_from_text(): no name found in first 5 lines")
        return ""

    # =========================================================================
    # LLM EXTRACTION — async, with retries
    # =========================================================================
    async def _extract_via_llm(self, resume_text: str) -> Optional[ResumeProfile]:
        """
        Attempt LLM-based extraction with retries.

        Sends the resume text to the configured LLM model using litellm,
        then parses the JSON response. Retries up to 2 additional times
        on failure (JSON parse errors or network issues).

        Called by: extract()
        Calls:     litellm.acompletion(), _extract_json(), _parse_response()

        Args:
            resume_text: The full raw resume text to send to the LLM

        Returns:
            ResumeProfile if successful, None if all attempts fail
        """
        max_retries = 2
        logger.debug(
            "_extract_via_llm(): starting LLM extraction, max_retries=%d, model=%s",
            max_retries, self.model,
        )

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "LLM attempt %d/%d: sending request to %s",
                    attempt + 1, max_retries + 1, self.model,
                )

                # Call the LLM via litellm (supports OpenAI, Anthropic, Ollama, etc.)
                response = await litellm.acompletion(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)}
                    ],
                    temperature=self.temperature,
                    max_tokens=4096,
                )

                # Extract the text content from the LLM response
                content = response.choices[0].message.content
                logger.debug(
                    "LLM response received: length=%d chars, first 200 chars=%r",
                    len(content) if content else 0,
                    (content[:200] if content else ""),
                )

                # Attempt to parse JSON from the LLM's text response
                data = self._extract_json(content)
                if data is None:
                    # JSON extraction failed — retry if attempts remain
                    if attempt < max_retries:
                        logger.warning(f"JSON extraction failed (attempt {attempt + 1}), retrying...")
                        continue
                    logger.warning("Could not extract valid JSON from LLM after retries")
                    return None

                logger.debug(
                    "LLM JSON parsed successfully: keys=%s", list(data.keys())
                )

                # Convert the raw dict into a structured ResumeProfile
                return self._parse_response(data, resume_text)

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"LLM extraction failed (attempt {attempt + 1}): {e}, retrying...")
                    continue
                logger.warning(f"LLM extraction failed after retries: {e}")
                return None

        return None

    # =========================================================================
    # MERGE — combines baseline + LLM results
    # =========================================================================
    def _merge_profiles(
        self, baseline: dict, llm_profile: Optional[ResumeProfile], raw_text: str
    ) -> ResumeProfile:
        """
        Merge regex baseline with LLM profile.

        Strategy:
          - LLM data takes priority for rich fields (skills, work history,
            projects, education, certifications, domain expertise).
          - Baseline fills in contact info gaps (email, phone, name, experience)
            that the LLM may have missed or hallucinated.

        Called by: extract()
        Calls:     nothing (leaf method, constructs ResumeProfile directly)

        Args:
            baseline:    dict from _extract_baseline() with name/email/phone/exp
            llm_profile: ResumeProfile from _extract_via_llm(), or None if failed
            raw_text:    original resume text (stored in the profile)

        Returns:
            Final merged ResumeProfile
        """
        logger.debug(
            "_merge_profiles(): llm_profile=%s",
            "present" if llm_profile else "None",
        )

        if llm_profile is None:
            # LLM completely failed — return baseline-only profile
            # Split the extracted name into first/last for the model
            name_parts = (baseline["name"] or "").split()
            first_name = name_parts[0] if name_parts else ""
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            logger.debug(
                "Falling back to baseline-only profile: first=%r, last=%r",
                first_name, last_name,
            )
            return ResumeProfile(
                first_name=first_name,
                last_name=last_name,
                email=baseline["email"],
                phone=baseline["phone"],
                total_experience_years=baseline["total_experience_years"],
                raw_text=raw_text,
            )

        # LLM succeeded — fill gaps with baseline data where LLM is empty
        if not llm_profile.email and baseline["email"]:
            logger.debug(
                "Filling LLM email gap with baseline: %r", baseline["email"]
            )
            llm_profile.email = baseline["email"]
        if not llm_profile.phone and baseline["phone"]:
            logger.debug(
                "Filling LLM phone gap with baseline: %r", baseline["phone"]
            )
            llm_profile.phone = baseline["phone"]
        if not llm_profile.full_name and baseline["name"]:
            # LLM didn't extract a name — use baseline's regex-extracted name
            name_parts = baseline["name"].split()
            llm_profile.first_name = name_parts[0] if name_parts else ""
            llm_profile.last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            logger.debug(
                "Filling LLM name gap with baseline: %r", baseline["name"]
            )
        if llm_profile.total_experience_years is None and baseline["total_experience_years"]:
            logger.debug(
                "Filling LLM experience gap with baseline: %s years",
                baseline["total_experience_years"],
            )
            llm_profile.total_experience_years = baseline["total_experience_years"]

        return llm_profile

    # =========================================================================
    # JSON EXTRACTION — handles messy LLM output
    # =========================================================================
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from LLM response, handling markdown code blocks,
        truncated responses, and other common formatting issues.

        The LLM is instructed to return pure JSON, but in practice it may:
          - Wrap it in ```json ... ``` markdown blocks
          - Include preamble text before the JSON
          - Truncate the response mid-JSON (token limit)
          - Add trailing commas (invalid JSON)

        This method tries multiple strategies in order of likelihood.

        Called by: _extract_via_llm()
        Calls:     _try_parse_json()

        Args:
            text: Raw LLM response string

        Returns:
            Parsed dict if successful, None otherwise
        """
        if not text:
            logger.debug("_extract_json(): received empty text, returning None")
            return None

        # Strategy 1: Direct parse — the ideal case (LLM returned clean JSON)
        try:
            result = json.loads(text)
            logger.debug("_extract_json(): direct json.loads() succeeded")
            return result
        except json.JSONDecodeError:
            logger.debug("_extract_json(): direct parse failed, trying alternatives")

        # Strategy 2: Extract from markdown code block ```json ... ``` or ``` ... ```
        json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_block:
            raw = json_block.group(1).strip()
            logger.debug(
                "_extract_json(): found markdown code block, length=%d", len(raw)
            )
            parsed = self._try_parse_json(raw)
            if parsed is not None:
                logger.debug("_extract_json(): markdown block parse succeeded")
                return parsed
            logger.debug("_extract_json(): markdown block parse failed")

        # Strategy 3: Find the first { ... } block (greedy, handles preamble text)
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            raw = brace_match.group(0)
            logger.debug(
                "_extract_json(): found brace block, length=%d", len(raw)
            )
            parsed = self._try_parse_json(raw)
            if parsed is not None:
                logger.debug("_extract_json(): brace block parse succeeded")
                return parsed
            logger.debug("_extract_json(): brace block parse failed")

        logger.debug("_extract_json(): all strategies exhausted, returning None")
        return None

    def _try_parse_json(self, raw: str) -> Optional[dict]:
        """
        Try to parse JSON, with fixups for common LLM issues.

        Attempts in order:
          1. Direct json.loads()
          2. Remove trailing commas before } or ] (common LLM mistake)
          3. Truncation recovery: try parsing up to each } from the end
             (handles responses cut off by token limits)

        Called by: _extract_json()
        Calls:     nothing (leaf method)

        Args:
            raw: A string that should contain JSON (possibly malformed)

        Returns:
            Parsed dict if any strategy works, None otherwise
        """
        # Attempt 1: Direct parse
        try:
            result = json.loads(raw)
            logger.debug("_try_parse_json(): direct parse succeeded")
            return result
        except json.JSONDecodeError:
            pass

        # Attempt 2: Remove trailing commas before } or ]
        # e.g., {"a": 1, "b": 2,} → {"a": 1, "b": 2}
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            result = json.loads(fixed)
            logger.debug("_try_parse_json(): trailing-comma fix succeeded")
            return result
        except json.JSONDecodeError:
            pass

        # Attempt 3: Truncation recovery
        # Walk backwards through the string, try parsing at each } position.
        # This handles cases where the LLM response was cut off mid-JSON.
        logger.debug(
            "_try_parse_json(): attempting truncation recovery (string length=%d)",
            len(raw),
        )
        for i in range(len(raw) - 1, 0, -1):
            if raw[i] == "}":
                try:
                    result = json.loads(raw[: i + 1])
                    logger.debug(
                        "_try_parse_json(): truncation recovery succeeded at position %d",
                        i,
                    )
                    return result
                except json.JSONDecodeError:
                    continue

        logger.debug("_try_parse_json(): all parse attempts failed")
        return None

    # =========================================================================
    # RESPONSE PARSING — converts LLM dict → ResumeProfile model
    # =========================================================================
    def _parse_response(self, data: dict, raw_text: str) -> ResumeProfile:
        """
        Parse LLM JSON response into ResumeProfile model.

        Handles various LLM output quirks:
          - Education may be list of strings OR list of dicts
          - Skills/certs may contain non-string items (filtered out)
          - total_experience_years may be missing (estimated from work history)

        Called by: _extract_via_llm()
        Calls:     _estimate_experience(), _estimate_experience_from_text()

        Args:
            data:     Parsed JSON dict from the LLM response
            raw_text: Original resume text (stored in profile)

        Returns:
            Fully populated ResumeProfile
        """
        logger.debug(
            "_parse_response(): parsing LLM data with %d keys", len(data)
        )

        # --- Parse work experiences ---
        # Each entry must be a dict; non-dict items are silently skipped
        work_experiences = [
            WorkExperience(
                company=exp.get("company") or "",
                title=exp.get("title") or "",
                duration_months=exp.get("duration_months"),
                start_year=exp.get("start_year"),
                end_year=exp.get("end_year"),
                is_current=exp.get("is_current", False),
                technologies=[t for t in (exp.get("technologies") or []) if isinstance(t, str)],
                responsibilities=[r for r in (exp.get("responsibilities") or []) if isinstance(r, str)],
                domain=exp.get("domain"),
            )
            for exp in data.get("work_experiences", [])
            if isinstance(exp, dict)
        ]
        logger.debug(
            "_parse_response(): parsed %d work experiences", len(work_experiences)
        )

        # --- Parse projects ---
        projects = [
            Project(
                name=proj.get("name") or "",
                description=proj.get("description") or "",
                technologies=[t for t in (proj.get("technologies") or []) if isinstance(t, str)],
                duration_months=proj.get("duration_months"),
                role=proj.get("role"),
            )
            for proj in data.get("projects", [])
            if isinstance(proj, dict)
        ]
        logger.debug("_parse_response(): parsed %d projects", len(projects))

        # --- Handle education ---
        # LLM may return list of strings OR list of dicts; normalize to strings
        raw_education = data.get("education") or []
        education = []
        for edu in raw_education:
            if isinstance(edu, str):
                education.append(edu)
            elif isinstance(edu, dict):
                # Convert dict to string format: "degree - institution - year"
                parts = [
                    edu.get("degree") or edu.get("name") or "",
                    edu.get("institution") or edu.get("school") or edu.get("university") or "",
                    str(edu.get("year") or ""),
                ]
                education.append(" - ".join(p for p in parts if p))
        logger.debug(
            "_parse_response(): parsed %d education entries", len(education)
        )

        # --- Filter skills to only valid strings ---
        # LLM sometimes returns None, numbers, or nested objects in the list
        raw_skills = data.get("skills") or []
        skills = [s for s in raw_skills if isinstance(s, str) and s.strip()]
        logger.debug(
            "_parse_response(): filtered skills: %d valid out of %d raw",
            len(skills), len(raw_skills),
        )

        # --- Filter certifications ---
        raw_certs = data.get("certifications") or []
        certifications = [c for c in raw_certs if isinstance(c, str) and c.strip()]
        logger.debug(
            "_parse_response(): filtered certifications: %d valid out of %d raw",
            len(certifications), len(raw_certs),
        )

        # --- Calculate total experience ---
        # Priority: LLM-provided value > estimated from work history > regex from text
        total_experience_years = data.get("total_experience_years")
        logger.debug(
            "_parse_response(): LLM total_experience_years=%s",
            total_experience_years,
        )

        if total_experience_years is None and work_experiences:
            # Estimate from work history durations/years
            total_experience_years = self._estimate_experience(work_experiences)
            logger.debug(
                "_parse_response(): estimated from work history=%s",
                total_experience_years,
            )
        if total_experience_years is None:
            # Last resort: regex patterns on raw text
            total_experience_years = self._estimate_experience_from_text(raw_text)
            logger.debug(
                "_parse_response(): estimated from raw text=%s",
                total_experience_years,
            )

        # --- Construct and return the final ResumeProfile ---
        return ResumeProfile(
            first_name=data.get("first_name") or "",
            middle_name=data.get("middle_name"),
            last_name=data.get("last_name") or "",
            email=data.get("email"),
            phone=data.get("phone"),
            location=data.get("location"),
            career_summary=data.get("career_summary") or "",
            skills=skills,
            total_experience_years=total_experience_years,
            work_experiences=work_experiences,
            projects=projects,
            education=education,
            certifications=certifications,
            domain_expertise=[d for d in (data.get("domain_expertise") or []) if isinstance(d, str)],
            raw_text=raw_text,
        )

    # =========================================================================
    # EXPERIENCE ESTIMATION — from structured work history
    # =========================================================================
    def _estimate_experience(self, work_experiences: list[WorkExperience]) -> Optional[float]:
        """
        Estimate total experience years from work history entries.

        Logic:
          - If an entry has duration_months, use it directly.
          - Otherwise, calculate from start_year to end_year (or current year
            if is_current is True).
          - Sum all months, convert to years (rounded to 1 decimal).

        Called by: _parse_response()
        Calls:     nothing (leaf method)

        Args:
            work_experiences: List of WorkExperience objects from LLM parsing

        Returns:
            Estimated years as float, or None if no data available
        """
        from datetime import datetime

        total_months = 0
        current_year = datetime.now().year
        logger.debug(
            "_estimate_experience(): calculating from %d work entries, current_year=%d",
            len(work_experiences), current_year,
        )

        for idx, exp in enumerate(work_experiences):
            if exp.duration_months:
                # Direct duration provided by LLM
                total_months += exp.duration_months
                logger.debug(
                    "  Entry %d (%s @ %s): using duration_months=%d",
                    idx, exp.title, exp.company, exp.duration_months,
                )
            elif exp.start_year:
                # Calculate from year range
                end = current_year if exp.is_current else (exp.end_year or current_year)
                months = (end - exp.start_year) * 12
                total_months += months
                logger.debug(
                    "  Entry %d (%s @ %s): calculated %d months (%d-%d)",
                    idx, exp.title, exp.company, months, exp.start_year, end,
                )
            else:
                logger.debug(
                    "  Entry %d (%s @ %s): no duration data available",
                    idx, exp.title, exp.company,
                )

        if total_months > 0:
            years = round(total_months / 12, 1)
            logger.debug(
                "_estimate_experience(): total_months=%d → %.1f years",
                total_months, years,
            )
            return years

        logger.debug("_estimate_experience(): no months accumulated, returning None")
        return None

    # =========================================================================
    # EXPERIENCE ESTIMATION — from raw text via regex
    # =========================================================================
    def _estimate_experience_from_text(self, raw_text: str) -> Optional[float]:
        """
        Fallback: extract experience years from raw resume text using patterns.

        Handles phrases like:
          - '10+ years of experience'
          - 'over 12 years in IT'
          - 'approximately 8 yrs experience'
          - 'experience of 15+ years'
          - 'decade of experience'

        Sanity check: only accepts values between 1 and 50 years.

        Called by: _extract_baseline(), _parse_response()
        Calls:     nothing (leaf method)

        Args:
            raw_text: Original resume text to scan

        Returns:
            Experience years as float, or None if no pattern matched
        """
        if not raw_text:
            return None

        logger.debug(
            "_estimate_experience_from_text(): scanning %d chars of text",
            len(raw_text),
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
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                years = float(match.group(1))
                if 1 <= years <= 50:  # sanity check
                    logger.debug(
                        "_estimate_experience_from_text(): pattern %r matched, years=%.1f",
                        pattern, years,
                    )
                    return years
                else:
                    logger.debug(
                        "_estimate_experience_from_text(): pattern %r matched but value %.1f outside 1-50 range",
                        pattern, years,
                    )

        # Handle "decade" / "plus decade" as a special case (= 10 years)
        if re.search(r"(?:over|plus)?\s*decade\s*(?:of\s+)?experience", raw_text, re.IGNORECASE):
            logger.debug(
                "_estimate_experience_from_text(): 'decade' pattern matched, returning 10.0"
            )
            return 10.0

        logger.debug("_estimate_experience_from_text(): no patterns matched")
        return None
