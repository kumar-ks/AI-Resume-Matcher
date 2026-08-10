"""
LangChain Chain: JD Extraction (Stage 1)
==========================================

Extracts structured job description requirements from raw text using:
  - ChatPromptTemplate with format instructions
  - ChatLiteLLMModel (our litellm wrapper)
  - PydanticOutputParser → JDExtractionOutput
  - Retry on parse failures (up to 2 retries)

USAGE:
    from matching_engine.chains.jd_chain import create_jd_chain

    chain = create_jd_chain(model="ollama/llama3", temperature=0.1)
    result = await chain.ainvoke({"jd_text": raw_jd_text})
    # result is a JDExtractionOutput (Pydantic model)

The caller (jd_understanding.py) converts JDExtractionOutput → JobDescription.
"""

import logging
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from matching_engine.chains.llm import ChatLiteLLMModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT MODEL — what the LLM should produce
# ─────────────────────────────────────────────────────────────────────────────


class JDSkillOutput(BaseModel):
    """A skill extracted from a job description."""
    name: str = Field(description="Name of the skill or technology")
    years_required: Optional[float] = Field(
        default=None, description="Minimum years of experience required, or null"
    )


class JDExtractionOutput(BaseModel):
    """Structured output from JD extraction LLM call."""
    title: str = Field(default="", description="Job title")
    must_have_skills: list[JDSkillOutput] = Field(
        default_factory=list,
        description="Skills that are mandatory for the role",
    )
    good_to_have_skills: list[JDSkillOutput] = Field(
        default_factory=list,
        description="Skills that are nice-to-have but not required",
    )
    experience_range_min: Optional[float] = Field(
        default=None, description="Minimum years of experience required"
    )
    experience_range_max: Optional[float] = Field(
        default=None, description="Maximum years of experience"
    )
    education: list[str] = Field(
        default_factory=list, description="Education/degree requirements"
    )
    domain_industry: list[str] = Field(
        default_factory=list, description="Relevant industry domains"
    )
    certifications: list[str] = Field(
        default_factory=list, description="Required or preferred certifications"
    )
    responsibilities: list[str] = Field(
        default_factory=list, description="Key job responsibilities"
    )
    location: Optional[str] = Field(
        default=None, description="Job location or null if remote/unspecified"
    )
    role_level: Optional[str] = Field(
        default=None,
        description="Role level: entry, mid, senior, lead, or principal",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

JD_SYSTEM_PROMPT = """You are an expert HR analyst. Analyze job descriptions and extract structured information.
Always respond with valid JSON matching the requested format. No markdown, no explanation — only JSON."""

JD_HUMAN_PROMPT = """Analyze the following job description and extract structured information.

{format_instructions}

Job Description:
---
{jd_text}
---

Return ONLY valid JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN FACTORY
# ─────────────────────────────────────────────────────────────────────────────


def create_jd_chain(
    model: str = "ollama/llama3",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 300,
    max_retries: int = 2,
    llm: Optional[ChatLiteLLMModel] = None,
):
    """
    Create a LangChain chain for JD extraction.

    The chain: prompt → LLM → PydanticOutputParser
    With retry: if parsing fails, retries the full chain up to max_retries times.

    Args:
        model: LiteLLM model identifier
        temperature: LLM temperature (lower = more deterministic)
        max_tokens: Max response tokens
        timeout: LLM timeout in seconds
        max_retries: Number of retries on parse failure
        llm: Optional pre-configured LLM instance (e.g., BifrostGatewayModel).
             If provided, model/temperature/max_tokens/timeout are ignored.

    Returns:
        A LangChain Runnable that accepts {"jd_text": str} and returns JDExtractionOutput
    """
    # Output parser with format instructions
    parser = PydanticOutputParser(pydantic_object=JDExtractionOutput)

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", JD_SYSTEM_PROMPT),
        ("human", JD_HUMAN_PROMPT),
    ]).partial(format_instructions=parser.get_format_instructions())

    # LLM — use provided instance or create a new one
    if llm is None:
        llm = ChatLiteLLMModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # Chain: prompt | llm | parser
    chain = prompt | llm | parser

    # Add retry on parse failures
    if max_retries > 0:
        chain = chain.with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=max_retries + 1,
            wait_exponential_jitter=True,
        )

    logger.debug(
        f"JD chain created: model={llm.model}, temperature={llm.temperature}, retries={max_retries}"
    )
    return chain
