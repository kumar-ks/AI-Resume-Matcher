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

import logging
from typing import Optional

import litellm

from matching_engine.models import (
    Project,
    ResumeProfile,
    WorkExperience,
)
from matching_engine.utils import (
    extract_json_from_llm_response,
    extract_name_from_text,
    extract_email_from_text,
    extract_phone_from_text,
    estimate_experience_from_text,
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

        # Step 4: Chunked extraction for work experiences (if LLM missed entries)
        # This is an ENHANCEMENT — it only runs if we detect the resume has more
        # experience entries than what the LLM extracted. It does NOT affect scoring.
        merged = await self._enhance_work_experiences(merged, resume_text)

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
        email = extract_email_from_text(text)
        logger.debug("Baseline email extracted: %r", email)

        # --- Extract phone ---
        # Pattern: optional country code, groups of 4-6 digits separated by
        # spaces or hyphens. Covers formats like +91 98765 43210, 123-4567-8901
        phone = extract_phone_from_text(text)
        logger.debug("Baseline phone extracted: %r", phone)

        # --- Extract name ---
        # Delegates to extract_name_from_text which scans the first few lines
        name = extract_name_from_text(text)
        logger.debug("Baseline name extracted: %r", name)

        # --- Extract experience ---
        # Delegates to estimate_experience_from_text which uses multiple
        # regex patterns to find phrases like "10+ years of experience"
        experience = estimate_experience_from_text(text)
        logger.debug("Baseline experience estimated: %s years", experience)

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "total_experience_years": experience,
        }

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
                    timeout=300,  # 5 min — allows for Ollama cold start + generation
                )

                # Extract the text content from the LLM response
                content = response.choices[0].message.content
                logger.debug(
                    "LLM response received: length=%d chars, first 200 chars=%r",
                    len(content) if content else 0,
                    (content[:200] if content else ""),
                )

                # Attempt to parse JSON from the LLM's text response
                data = extract_json_from_llm_response(content)
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
            total_experience_years = estimate_experience_from_text(raw_text)
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
    # CHUNKED EXTRACTION — supplements work experiences for large resumes
    # =========================================================================

    # Prompt specifically for extracting ONLY work experience entries.
    # Shorter output = less likely to be truncated by the LLM.
    EXPERIENCE_CHUNK_PROMPT = """Extract ALL work experiences from this resume text. Return a JSON array.

Each entry must have:
{{
    "company": "company name",
    "title": "job title",
    "start_year": number or null,
    "end_year": number or null,
    "is_current": boolean,
    "technologies": ["tech1", "tech2"],
    "responsibilities": ["resp1", "resp2", "resp3"]
}}

Resume text (EXPERIENCE section only):
---
{experience_text}
---

Return ONLY a JSON array [...], no markdown, no explanation."""

    async def _enhance_work_experiences(
        self, profile: ResumeProfile, raw_text: str
    ) -> ResumeProfile:
        """
        Enhance work experiences by sending ONLY the experience section to the LLM.

        This is a SUPPLEMENTARY step that runs AFTER the main extraction.
        It only triggers if the resume appears to have more experience entries
        than what was initially extracted (detected via year-pattern counting).

        The existing profile data (name, skills, scoring fields) is NEVER modified.
        Only work_experiences[] may be replaced with a more complete list.

        Called by: extract() as Step 4
        Does NOT affect: scoring, semantic matching, or any pipeline stage

        Args:
            profile: The merged ResumeProfile from Steps 1-3
            raw_text: The full raw resume text

        Returns:
            The same profile with potentially more complete work_experiences
        """
        import re

        # Count how many experience entries the resume likely has
        # (by counting year patterns like "2020", "2023 - Present", etc.)
        year_patterns = re.findall(r"(19|20)\d{2}", raw_text)
        estimated_entries = len(year_patterns) // 2  # Each entry has ~2 years (start + end)

        current_entries = len(profile.work_experiences)

        # Only run chunked extraction if we're clearly missing entries
        if current_entries >= estimated_entries or estimated_entries <= 2:
            logger.debug(
                f"Chunked extraction skipped: have {current_entries} entries, "
                f"estimated {estimated_entries} in text"
            )
            return profile

        logger.info(
            f"Chunked extraction triggered: have {current_entries} entries, "
            f"estimated {estimated_entries} in text. Extracting experience section..."
        )

        # Extract the experience section from the raw text
        experience_text = self._extract_experience_section(raw_text)
        if not experience_text or len(experience_text) < 50:
            logger.debug("Could not isolate experience section, skipping chunked extraction")
            return profile

        # Send just the experience section to the LLM (smaller input = complete output)
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.EXPERIENCE_CHUNK_PROMPT.format(
                            experience_text=experience_text
                        ),
                    }
                ],
                temperature=self.temperature,
                max_tokens=4096,
                timeout=300,
            )

            content = response.choices[0].message.content
            data = extract_json_from_llm_response(content)

            # The response should be a list (JSON array) of experience dicts
            if data is None:
                # Try parsing as array directly
                import json
                try:
                    # Handle case where response is a raw JSON array
                    if content and content.strip().startswith("["):
                        data = json.loads(content.strip())
                except (json.JSONDecodeError, TypeError):
                    pass

            if data is None:
                logger.warning("Chunked extraction failed to produce valid JSON")
                return profile

            # If data is a dict with a "work_experiences" key, extract it
            if isinstance(data, dict):
                entries = data.get("work_experiences", data.get("experiences", []))
            elif isinstance(data, list):
                entries = data
            else:
                return profile

            # Parse the entries into WorkExperience objects
            new_experiences = []
            for exp in entries:
                if not isinstance(exp, dict):
                    continue
                try:
                    new_experiences.append(
                        WorkExperience(
                            company=exp.get("company") or "",
                            title=exp.get("title") or "",
                            duration_months=exp.get("duration_months"),
                            start_year=exp.get("start_year"),
                            end_year=exp.get("end_year"),
                            is_current=exp.get("is_current", False),
                            technologies=[
                                t for t in (exp.get("technologies") or [])
                                if isinstance(t, str)
                            ],
                            responsibilities=[
                                r for r in (exp.get("responsibilities") or [])
                                if isinstance(r, str)
                            ],
                            domain=exp.get("domain"),
                        )
                    )
                except Exception:
                    continue

            # Only replace if we got MORE entries than before
            if len(new_experiences) > current_entries:
                logger.info(
                    f"Chunked extraction succeeded: {current_entries} → {len(new_experiences)} entries"
                )
                profile.work_experiences = new_experiences
            else:
                logger.debug(
                    f"Chunked extraction got {len(new_experiences)} entries "
                    f"(not more than existing {current_entries}), keeping original"
                )

        except Exception as e:
            logger.warning(f"Chunked experience extraction failed: {e}")

        return profile

    def _extract_experience_section(self, raw_text: str) -> str:
        """
        Extract just the EXPERIENCE/WORK HISTORY section from raw resume text.

        Uses section header detection to isolate the experience portion,
        reducing the input size for the chunked LLM call.

        Returns:
            The experience section text, or empty string if not found.
        """
        import re

        lines = raw_text.split("\n")
        experience_start = None
        experience_end = None

        # Find where experience section starts
        experience_headers = [
            "experience", "professional experience", "work experience",
            "employment history", "work history", "career history",
        ]

        for i, line in enumerate(lines):
            normalized = re.sub(r"\s+", " ", line).strip().lower()
            if any(normalized == h or normalized.startswith(h) for h in experience_headers):
                experience_start = i
                break

        if experience_start is None:
            # Fallback: look for first year pattern as start of experience
            for i, line in enumerate(lines):
                if re.search(r"(19|20)\d{2}\s*[-–—]\s*(present|(19|20)\d{2})", line, re.IGNORECASE):
                    experience_start = max(0, i - 1)
                    break

        if experience_start is None:
            return ""

        # Find where experience section ends (next major section header)
        end_headers = [
            "education", "certifications", "projects", "publications",
            "awards", "skills", "references", "interests", "hobbies",
        ]

        for i in range(experience_start + 1, len(lines)):
            normalized = re.sub(r"\s+", " ", lines[i]).strip().lower()
            if any(normalized == h or normalized.startswith(h) for h in end_headers):
                experience_end = i
                break

        if experience_end is None:
            experience_end = len(lines)

        experience_text = "\n".join(lines[experience_start:experience_end])
        logger.debug(
            f"Extracted experience section: lines {experience_start}-{experience_end}, "
            f"{len(experience_text)} chars"
        )
        return experience_text
