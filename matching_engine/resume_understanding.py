"""
Stage 2: Resume Understanding

Extracts structured candidate information from raw resume text using a
combination of regex-based baseline extraction and LangChain-powered deep parsing.

=== IMPLEMENTATION ===

Uses a LangChain chain (prompt → ChatLiteLLMModel → PydanticOutputParser) for
structured extraction with automatic retry on parse failures. Baseline regex
extraction provides a safety net when the LLM fails.

=== CALL HIERARCHY & FLOW ===

    extract() [PUBLIC - main entry point, called by pipeline.py]
        │
        ├── _extract_baseline(text)     [regex - always succeeds]
        │
        ├── _extract_via_chain(text)    [LangChain chain with retry]
        │       └── create_resume_chain() → ainvoke()
        │       └── _convert_chain_output()
        │
        ├── _merge_profiles(baseline, llm_profile, raw_text)
        │
        └── _enhance_work_experiences(profile, raw_text)
                └── create_experience_chunk_chain() → ainvoke()

=== DATA FLOW ===

1. extract() receives raw resume text (string from PDF/DOCX/text).
2. _extract_baseline() uses regex to pull name, email, phone, experience
   — this ALWAYS succeeds and provides a safety net.
3. _extract_via_chain() invokes the LangChain chain to produce a
   ResumeExtractionOutput — the chain handles retries internally.
4. _merge_profiles() combines both: LLM data takes priority for rich fields
   (skills, work history, projects), baseline fills contact-info gaps.
5. _enhance_work_experiences() runs a focused chain on just the experience
   section if the initial extraction appears incomplete.
"""

import logging
import re
from datetime import datetime
from typing import Any, Optional

from matching_engine.chains.factory import get_llm_for_stage
from matching_engine.chains.resume_chain import (
    ResumeExtractionOutput,
    WorkExperienceOutput,
    create_experience_chunk_chain,
    create_resume_chain,
)
from matching_engine.models import (
    Project,
    ResumeProfile,
    WorkExperience,
)
from matching_engine.observability import (
    create_generation,
    end_generation,
)
from matching_engine.utils import (
    estimate_experience_from_text,
    extract_email_from_text,
    extract_name_from_text,
    extract_phone_from_text,
)

logger = logging.getLogger(__name__)


class ResumeUnderstanding:
    """
    Parses raw resume text into a structured ResumeProfile model.

    Uses a LangChain chain with PydanticOutputParser for deep extraction,
    combined with regex-based baseline extraction for reliability.

    Two-pass strategy:
      Pass 1 (baseline): Regex extraction — fast, deterministic, always works.
      Pass 2 (LLM chain): Deep extraction via LangChain — richer data, may fail.
      Merge:             LLM wins for rich fields; baseline fills gaps.
    """

    def __init__(self, model: str = "ollama/llama2", temperature: float = 0.1):
        """
        Initialize Resume Understanding stage.

        Args:
            model: LiteLLM model identifier
            temperature: LLM temperature for extraction (low = deterministic)
        """
        self.model = model
        self.temperature = temperature
        self._trace_parent: Optional[Any] = None

        # Create LangChain chains (uses factory for PII routing)
        resume_llm = get_llm_for_stage("resume", model=model, temperature=temperature)
        self._chain = create_resume_chain(
            model=model,
            temperature=temperature,
            max_tokens=4096,
            timeout=300,
            max_retries=2,
            llm=resume_llm,
        )
        self._experience_chain = create_experience_chunk_chain(
            model=model,
            temperature=temperature,
            max_tokens=4096,
            timeout=300,
            max_retries=1,
            llm=resume_llm,
        )

        logger.debug(f"ResumeUnderstanding initialized: model={model}, temperature={temperature}")

    # =========================================================================
    # PUBLIC METHOD — called by pipeline.py
    # =========================================================================
    async def extract(self, resume_text: str, trace_parent: Optional[Any] = None) -> ResumeProfile:
        """
        Extract structured profile from raw resume text.

        This is the MAIN ENTRY POINT for Stage 2. It orchestrates:
          1. _extract_baseline()   → regex-based contact info & experience
          2. _extract_via_chain()  → LangChain chain extraction (may fail)
          3. _merge_profiles()     → combine both, LLM priority
          4. _enhance_work_experiences() → focused chain for experience section

        Args:
            resume_text: Raw resume text (from PDF/DOCX/plain text)
            trace_parent: Optional LangFuse span/trace for observability

        Returns:
            ResumeProfile with all extracted fields populated
        """
        logger.info("Stage 2: Extracting resume information")
        parent = trace_parent or self._trace_parent

        if not resume_text.strip():
            logger.warning("Empty resume text provided")
            return ResumeProfile(raw_text=resume_text)

        # Step 1: Regex-based extraction (always succeeds)
        baseline = self._extract_baseline(resume_text)

        # Step 2: LangChain chain extraction (may fail)
        llm_profile = await self._extract_via_chain(resume_text, trace_parent=parent)

        # Step 3: Merge — LLM data priority, baseline fills gaps
        merged = self._merge_profiles(baseline, llm_profile, resume_text)

        # Step 4: Enhance work experiences if extraction seems incomplete
        merged = await self._enhance_work_experiences(merged, resume_text, trace_parent=parent)

        return merged

    # =========================================================================
    # BASELINE EXTRACTION — regex-based, deterministic
    # =========================================================================
    def _extract_baseline(self, text: str) -> dict:
        """
        Extract basic contact info and experience using regex patterns.
        This always produces results regardless of LLM availability.
        """
        email = extract_email_from_text(text)
        phone = extract_phone_from_text(text)
        name = extract_name_from_text(text)
        experience = estimate_experience_from_text(text)

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "total_experience_years": experience,
        }

    # =========================================================================
    # LANGCHAIN CHAIN EXTRACTION
    # =========================================================================
    async def _extract_via_chain(
        self, resume_text: str, trace_parent: Optional[Any] = None
    ) -> Optional[ResumeProfile]:
        """
        Extract resume data via LangChain chain with retry.

        The chain (prompt → LLM → PydanticOutputParser) handles retries
        internally. If all attempts fail, returns None.

        Args:
            resume_text: Raw resume text
            trace_parent: LangFuse span for observability

        Returns:
            ResumeProfile if successful, None if chain fails
        """
        generation = create_generation(
            parent=trace_parent,
            name="resume-extraction-chain",
            model=self.model,
            input_data={"resume_text_length": len(resume_text)},
            model_parameters={"temperature": self.temperature, "max_tokens": 4096},
            metadata={"method": "langchain_chain"},
        )

        try:
            result: ResumeExtractionOutput = await self._chain.ainvoke(
                {"resume_text": resume_text}
            )

            logger.debug(
                f"Chain returned: name='{result.first_name} {result.last_name}', "
                f"skills={len(result.skills)}, experiences={len(result.work_experiences)}"
            )

            end_generation(generation, output=result.model_dump())

            return self._convert_chain_output(result, resume_text)

        except Exception as e:
            logger.warning(f"Resume extraction chain failed: {e}")
            end_generation(generation, level="ERROR", status_message=str(e))
            return None

    def _convert_chain_output(
        self, output: ResumeExtractionOutput, raw_text: str
    ) -> ResumeProfile:
        """Convert LangChain chain output to pipeline's ResumeProfile model."""

        work_experiences = [
            WorkExperience(
                company=exp.company,
                title=exp.title,
                duration_months=exp.duration_months,
                start_year=exp.start_year,
                end_year=exp.end_year,
                is_current=exp.is_current,
                technologies=exp.technologies,
                responsibilities=exp.responsibilities,
                domain=exp.domain,
            )
            for exp in output.work_experiences
        ]

        projects = [
            Project(
                name=proj.name,
                description=proj.description,
                technologies=proj.technologies,
                duration_months=proj.duration_months,
                role=proj.role,
            )
            for proj in output.projects
        ]

        # Calculate experience if not provided
        total_exp = output.total_experience_years
        if total_exp is None and work_experiences:
            total_exp = self._estimate_experience(work_experiences)
        if total_exp is None:
            total_exp = estimate_experience_from_text(raw_text)

        return ResumeProfile(
            first_name=output.first_name,
            middle_name=output.middle_name,
            last_name=output.last_name,
            email=output.email,
            phone=output.phone,
            location=output.location,
            career_summary=output.career_summary,
            skills=output.skills,
            total_experience_years=total_exp,
            work_experiences=work_experiences,
            projects=projects,
            education=output.education,
            certifications=output.certifications,
            domain_expertise=output.domain_expertise,
            raw_text=raw_text,
        )

    # =========================================================================
    # MERGE — combines baseline + LLM results
    # =========================================================================
    def _merge_profiles(
        self, baseline: dict, llm_profile: Optional[ResumeProfile], raw_text: str
    ) -> ResumeProfile:
        """
        Merge regex baseline with LLM profile.

        Strategy:
          - LLM data takes priority for rich fields (skills, work history, projects).
          - Baseline fills in contact info gaps (email, phone, name, experience).
        """
        if llm_profile is None:
            # LLM completely failed — return baseline-only profile
            name_parts = (baseline["name"] or "").split()
            first_name = name_parts[0] if name_parts else ""
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            return ResumeProfile(
                first_name=first_name,
                last_name=last_name,
                email=baseline["email"],
                phone=baseline["phone"],
                total_experience_years=baseline["total_experience_years"],
                raw_text=raw_text,
            )

        # LLM succeeded — fill gaps with baseline data
        if not llm_profile.email and baseline["email"]:
            llm_profile.email = baseline["email"]
        if not llm_profile.phone and baseline["phone"]:
            llm_profile.phone = baseline["phone"]
        if not llm_profile.full_name and baseline["name"]:
            name_parts = baseline["name"].split()
            llm_profile.first_name = name_parts[0] if name_parts else ""
            llm_profile.last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        if llm_profile.total_experience_years is None and baseline["total_experience_years"]:
            llm_profile.total_experience_years = baseline["total_experience_years"]

        return llm_profile

    # =========================================================================
    # EXPERIENCE ESTIMATION
    # =========================================================================
    def _estimate_experience(self, work_experiences: list[WorkExperience]) -> Optional[float]:
        """Estimate total experience years from work history entries."""
        total_months = 0
        current_year = datetime.now().year

        for exp in work_experiences:
            if exp.duration_months:
                total_months += exp.duration_months
            elif exp.start_year:
                end = current_year if exp.is_current else (exp.end_year or current_year)
                total_months += (end - exp.start_year) * 12

        if total_months > 0:
            return round(total_months / 12, 1)
        return None

    # =========================================================================
    # CHUNKED EXTRACTION — supplements work experiences for large resumes
    # =========================================================================
    async def _enhance_work_experiences(
        self, profile: ResumeProfile, raw_text: str, trace_parent: Optional[Any] = None
    ) -> ResumeProfile:
        """
        Enhance work experiences by extracting ONLY the experience section via a focused chain.

        Only triggers if the resume appears to have more experience entries
        than what was initially extracted (detected via year-pattern counting).
        """
        # Count how many experience entries the resume likely has
        year_patterns = re.findall(r"(19|20)\d{2}", raw_text)
        estimated_entries = len(year_patterns) // 2
        current_entries = len(profile.work_experiences)

        if current_entries >= estimated_entries or estimated_entries <= 2:
            return profile

        logger.info(
            f"Chunked extraction triggered: have {current_entries} entries, "
            f"estimated {estimated_entries} in text."
        )

        # Extract the experience section from raw text
        experience_text = self._extract_experience_section(raw_text)
        if not experience_text or len(experience_text) < 50:
            return profile

        generation = create_generation(
            parent=trace_parent,
            name="resume-chunked-experience-chain",
            model=self.model,
            input_data={"experience_text_length": len(experience_text)},
            metadata={"current_entries": current_entries, "estimated_entries": estimated_entries},
        )

        try:
            result = await self._experience_chain.ainvoke(
                {"experience_text": experience_text}
            )

            end_generation(generation, output={"entries_found": len(result.work_experiences)})

            # Convert to WorkExperience objects
            new_experiences = [
                WorkExperience(
                    company=exp.company,
                    title=exp.title,
                    duration_months=exp.duration_months,
                    start_year=exp.start_year,
                    end_year=exp.end_year,
                    is_current=exp.is_current,
                    technologies=exp.technologies,
                    responsibilities=exp.responsibilities,
                    domain=exp.domain,
                )
                for exp in result.work_experiences
            ]

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
            end_generation(generation, level="ERROR", status_message=str(e))
            logger.warning(f"Chunked experience extraction failed: {e}")

        return profile

    def _extract_experience_section(self, raw_text: str) -> str:
        """
        Extract just the EXPERIENCE/WORK HISTORY section from raw resume text.
        Uses section header detection to isolate the experience portion.
        """
        lines = raw_text.split("\n")
        experience_start = None
        experience_end = None

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
            for i, line in enumerate(lines):
                if re.search(r"(19|20)\d{2}\s*[-–—]\s*(present|(19|20)\d{2})", line, re.IGNORECASE):
                    experience_start = max(0, i - 1)
                    break

        if experience_start is None:
            return ""

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

        return "\n".join(lines[experience_start:experience_end])
