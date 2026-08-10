"""
LangChain Chain: Resume Extraction (Stage 2)
==============================================

Extracts structured resume profile from raw text using:
  - ChatPromptTemplate with format instructions
  - ChatLiteLLMModel (our litellm wrapper)
  - PydanticOutputParser → ResumeExtractionOutput
  - Retry on parse failures (up to 2 retries)

USAGE:
    from matching_engine.chains.resume_chain import create_resume_chain

    chain = create_resume_chain(model="ollama/llama3", temperature=0.1)
    result = await chain.ainvoke({"resume_text": raw_resume_text})
    # result is a ResumeExtractionOutput (Pydantic model)

The caller (resume_understanding.py) converts ResumeExtractionOutput → ResumeProfile.
"""

import logging
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from matching_engine.chains.llm import ChatLiteLLMModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT MODELS
# ─────────────────────────────────────────────────────────────────────────────


class WorkExperienceOutput(BaseModel):
    """A work experience entry extracted from a resume."""
    company: str = Field(default="", description="Company name")
    title: str = Field(default="", description="Job title")
    duration_months: Optional[int] = Field(
        default=None, description="Duration in months, or null"
    )
    start_year: Optional[int] = Field(default=None, description="Start year")
    end_year: Optional[int] = Field(
        default=None, description="End year, or null if current"
    )
    is_current: bool = Field(default=False, description="Whether this is the current role")
    technologies: list[str] = Field(
        default_factory=list, description="Technologies used in this role"
    )
    responsibilities: list[str] = Field(
        default_factory=list, description="Key responsibilities"
    )
    domain: Optional[str] = Field(
        default=None, description="Industry domain, or null"
    )


class ProjectOutput(BaseModel):
    """A project extracted from a resume."""
    name: str = Field(default="", description="Project name")
    description: str = Field(default="", description="Brief project description")
    technologies: list[str] = Field(
        default_factory=list, description="Technologies used"
    )
    duration_months: Optional[int] = Field(
        default=None, description="Duration in months"
    )
    role: Optional[str] = Field(default=None, description="Role in the project")


class ResumeExtractionOutput(BaseModel):
    """Structured output from resume extraction LLM call."""
    first_name: str = Field(default="", description="Candidate's first name")
    middle_name: Optional[str] = Field(default=None, description="Middle name or null")
    last_name: str = Field(default="", description="Candidate's last name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="Location")
    career_summary: str = Field(
        default="", description="Brief career summary in 2-3 sentences"
    )
    skills: list[str] = Field(
        default_factory=list, description="All technical and soft skills"
    )
    total_experience_years: Optional[float] = Field(
        default=None, description="Total years of professional experience"
    )
    work_experiences: list[WorkExperienceOutput] = Field(
        default_factory=list, description="Work experience entries"
    )
    projects: list[ProjectOutput] = Field(
        default_factory=list, description="Notable projects"
    )
    education: list[str] = Field(
        default_factory=list,
        description="Education entries as 'degree - institution - year'",
    )
    certifications: list[str] = Field(
        default_factory=list, description="Professional certifications"
    )
    domain_expertise: list[str] = Field(
        default_factory=list, description="Industry domains the candidate has worked in"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

RESUME_SYSTEM_PROMPT = """You are an expert resume parser. Extract structured information from resumes.
Always respond with valid JSON matching the requested format. No markdown, no explanation — only JSON."""

RESUME_HUMAN_PROMPT = """Analyze the following resume and extract structured information.

{format_instructions}

Resume:
---
{resume_text}
---

Return ONLY valid JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN FACTORY
# ─────────────────────────────────────────────────────────────────────────────


def create_resume_chain(
    model: str = "ollama/llama3",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 300,
    max_retries: int = 2,
    llm: Optional[ChatLiteLLMModel] = None,
):
    """
    Create a LangChain chain for resume extraction.

    The chain: prompt → LLM → PydanticOutputParser
    With retry: if parsing fails, retries the full chain up to max_retries times.

    Args:
        model: LiteLLM model identifier
        temperature: LLM temperature (lower = more deterministic)
        max_tokens: Max response tokens
        timeout: LLM timeout in seconds
        max_retries: Number of retries on parse failure
        llm: Optional pre-configured LLM instance (e.g., BifrostGatewayModel)

    Returns:
        A LangChain Runnable that accepts {"resume_text": str} and returns ResumeExtractionOutput
    """
    parser = PydanticOutputParser(pydantic_object=ResumeExtractionOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RESUME_SYSTEM_PROMPT),
        ("human", RESUME_HUMAN_PROMPT),
    ]).partial(format_instructions=parser.get_format_instructions())

    if llm is None:
        llm = ChatLiteLLMModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    chain = prompt | llm | parser

    if max_retries > 0:
        chain = chain.with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=max_retries + 1,
            wait_exponential_jitter=True,
        )

    logger.debug(
        f"Resume chain created: model={llm.model}, temperature={llm.temperature}, retries={max_retries}"
    )
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE CHUNK CHAIN (for _enhance_work_experiences)
# ─────────────────────────────────────────────────────────────────────────────

EXPERIENCE_CHUNK_SYSTEM_PROMPT = """You are an expert resume parser. Extract ONLY work experience entries.
Always respond with valid JSON matching the requested format. No markdown — only JSON."""

EXPERIENCE_CHUNK_HUMAN_PROMPT = """Extract ALL work experiences from this resume text.

{format_instructions}

Resume text (EXPERIENCE section only):
---
{experience_text}
---

Return ONLY valid JSON."""


class ExperienceChunkOutput(BaseModel):
    """Output for chunked experience extraction."""
    work_experiences: list[WorkExperienceOutput] = Field(
        default_factory=list, description="All work experience entries"
    )


def create_experience_chunk_chain(
    model: str = "ollama/llama3",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 300,
    max_retries: int = 1,
    llm: Optional[ChatLiteLLMModel] = None,
):
    """
    Create a chain specifically for extracting work experiences from a text chunk.

    Used by resume_understanding.py's _enhance_work_experiences() method.

    Args:
        llm: Optional pre-configured LLM instance

    Returns:
        A Runnable that accepts {"experience_text": str} and returns ExperienceChunkOutput
    """
    parser = PydanticOutputParser(pydantic_object=ExperienceChunkOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPERIENCE_CHUNK_SYSTEM_PROMPT),
        ("human", EXPERIENCE_CHUNK_HUMAN_PROMPT),
    ]).partial(format_instructions=parser.get_format_instructions())

    if llm is None:
        llm = ChatLiteLLMModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    chain = prompt | llm | parser

    if max_retries > 0:
        chain = chain.with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=max_retries + 1,
            wait_exponential_jitter=True,
        )

    return chain
