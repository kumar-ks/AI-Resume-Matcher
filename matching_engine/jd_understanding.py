"""
Stage 1: JD Understanding
==========================

PURPOSE:
    Extracts structured requirements from raw job description text using an LLM.
    Identifies must-have skills, good-to-have skills, experience range,
    education requirements, domain/industry, and certifications.

IMPLEMENTATION:
    Uses a LangChain chain (prompt → LLM → PydanticOutputParser) for structured
    extraction with automatic retry on parse failures. Falls back to keyword-based
    extraction if the chain fails entirely.

CALL HIERARCHY:
    Called by:
        - pipeline.py → calls JDUnderstanding.extract() as Stage 1 of the matching pipeline
    Calls internally:
        - create_jd_chain() → LangChain chain (prompt | ChatLiteLLMModel | PydanticOutputParser)
        - _convert_chain_output() → maps JDExtractionOutput → JobDescription
        - _parse_role_level() → maps string role level to ExperienceLevel enum
        - _fallback_extraction() → provides minimal extraction when chain fails

RETURNS:
    JobDescription model with all extracted fields populated (title, skills,
    experience range, education, domain, certifications, responsibilities, etc.)
"""

import logging
from typing import Any, Optional

from matching_engine.chains.factory import get_llm_for_stage
from matching_engine.chains.jd_chain import JDExtractionOutput, create_jd_chain
from matching_engine.models import (
    ExperienceLevel,
    JobDescription,
    Skill,
    SkillCategory,
)
from matching_engine.observability import (
    create_generation,
    end_generation,
)

logger = logging.getLogger(__name__)


class JDUnderstanding:
    """
    Parses raw job description text into a structured JobDescription model.

    Uses a LangChain chain with PydanticOutputParser to extract key requirements
    including skills (must-have vs good-to-have), experience range, education,
    domain, and certifications.

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
        self._trace_parent: Optional[Any] = None  # Set by pipeline for LangFuse tracing

        # Create the LangChain chain (prompt → LLM → PydanticOutputParser with retry)
        # Uses factory to get the right LLM (local or Bi-Frost gateway)
        llm = get_llm_for_stage("jd", model=model, temperature=temperature)
        self._chain = create_jd_chain(
            model=model,
            temperature=temperature,
            max_tokens=4096,
            timeout=300,
            max_retries=2,
            llm=llm,
        )

        logger.debug(f"JDUnderstanding initialized with model={model}, temperature={temperature}")

    async def extract(self, jd_text: str, trace_parent: Optional[Any] = None) -> JobDescription:
        """
        Extract structured requirements from raw JD text.

        Called by: pipeline.py as Stage 1 of the matching pipeline.

        Flow:
            1. Validate input text is not empty
            2. Invoke LangChain chain with jd_text (includes retry on parse failure)
            3. Convert chain output (JDExtractionOutput) → JobDescription
            4. On failure, fall back to keyword-based extraction

        Args:
            jd_text: Raw job description text (from PDF/DOCX/plain text)
            trace_parent: Optional LangFuse span/trace for observability

        Returns:
            JobDescription with all extracted fields populated
        """
        logger.info("Stage 1: Extracting JD requirements")
        logger.debug(f"JD text length: {len(jd_text)} characters")

        # Use explicitly passed parent, or fall back to instance-level parent
        parent = trace_parent or self._trace_parent

        # Guard clause: if JD text is empty/whitespace, return empty model
        if not jd_text.strip():
            logger.warning("Empty JD text provided")
            return JobDescription(raw_text=jd_text)

        # Create LangFuse generation span for observability
        generation = create_generation(
            parent=parent,
            name="jd-extraction-chain",
            model=self.model,
            input_data={"jd_text_length": len(jd_text)},
            model_parameters={"temperature": self.temperature, "max_tokens": 4096},
            metadata={"method": "langchain_chain"},
        )

        try:
            # Invoke the LangChain chain (prompt → LLM → PydanticOutputParser)
            # The chain handles retries internally on parse failures
            logger.debug(f"Invoking JD chain with model: {self.model}")
            result: JDExtractionOutput = await self._chain.ainvoke({"jd_text": jd_text})

            logger.debug(f"Chain returned: title='{result.title}', skills={len(result.must_have_skills)}")

            # End generation with success
            end_generation(
                generation,
                output=result.model_dump(),
            )

            # Convert chain output to pipeline's JobDescription model
            return self._convert_chain_output(result, jd_text)

        except Exception as e:
            logger.error(f"JD extraction chain failed: {e}")
            end_generation(generation, level="ERROR", status_message=str(e))
            return self._fallback_extraction(jd_text)

    def _convert_chain_output(self, output: JDExtractionOutput, raw_text: str) -> JobDescription:
        """
        Convert LangChain chain output (JDExtractionOutput) to pipeline's JobDescription model.

        Maps the Pydantic output model from the chain into the full JobDescription
        model used by downstream pipeline stages.

        Args:
            output: Parsed JDExtractionOutput from the chain
            raw_text: Original raw JD text (preserved in the model)

        Returns:
            Fully populated JobDescription model
        """
        logger.debug(f"Converting chain output: title='{output.title}'")

        # Convert skill output models to pipeline Skill objects
        must_have = [
            Skill(
                name=s.name,
                category=SkillCategory.MUST_HAVE,
                years_required=s.years_required,
            )
            for s in output.must_have_skills
        ]

        good_to_have = [
            Skill(
                name=s.name,
                category=SkillCategory.GOOD_TO_HAVE,
                years_required=s.years_required,
            )
            for s in output.good_to_have_skills
        ]

        role_level = self._parse_role_level(output.role_level)

        return JobDescription(
            title=output.title,
            must_have_skills=must_have,
            good_to_have_skills=good_to_have,
            experience_range_min=output.experience_range_min,
            experience_range_max=output.experience_range_max,
            education=output.education,
            domain_industry=output.domain_industry,
            certifications=output.certifications,
            responsibilities=output.responsibilities,
            location=output.location,
            role_level=role_level,
            raw_text=raw_text,
        )

    def _parse_role_level(self, level: Optional[str]) -> Optional[ExperienceLevel]:
        """
        Map string role level to ExperienceLevel enum.

        Args:
            level: String like "entry", "mid", "senior", "lead", "principal"

        Returns:
            Corresponding ExperienceLevel enum value, or None if not recognized
        """
        if not level:
            return None

        mapping = {
            "entry": ExperienceLevel.ENTRY,
            "mid": ExperienceLevel.MID,
            "senior": ExperienceLevel.SENIOR,
            "lead": ExperienceLevel.LEAD,
            "principal": ExperienceLevel.PRINCIPAL,
        }
        return mapping.get(level.lower())

    def _fallback_extraction(self, jd_text: str) -> JobDescription:
        """
        Basic keyword-based extraction when the LangChain chain fails.

        Provides a minimal extraction using simple heuristics.
        The raw_text is preserved so downstream stages can still
        use the full text for semantic matching.

        Args:
            jd_text: Original raw JD text

        Returns:
            Minimal JobDescription with only raw_text populated
        """
        logger.info("Using fallback keyword extraction for JD")
        return JobDescription(raw_text=jd_text)
