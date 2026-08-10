"""
LangChain Chain: Explainability (Stage 5)
============================================

Generates human-readable explanations for match scores using:
  - ChatPromptTemplate with scoring data placeholders
  - ChatLiteLLMModel (our litellm wrapper)
  - PydanticOutputParser → ExplainOutput
  - Retry on parse failures

USAGE:
    from matching_engine.chains.explain_chain import create_explain_chain

    chain = create_explain_chain(model="ollama/llama3", temperature=0.3)
    result = await chain.ainvoke({
        "job_title": "Software Engineer",
        "candidate_name": "John Doe",
        "score": 75,
        "must_have_skills": "Python, AWS, Docker",
        "candidate_skills": "Python, Kubernetes, Terraform",
        "exp_required": "5-10 years",
        "candidate_exp": 8,
        "must_have_score": 0.82,
        "exp_score": 0.75,
        "skills_depth": 0.68,
        "project_score": 0.45,
        "recency_score": 0.90,
    })
    # result is an ExplainOutput (Pydantic model)

The caller (explainability.py) converts ExplainOutput → ExplainabilityReport.
"""

import logging
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from matching_engine.chains.llm import ChatLiteLLMModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT MODEL
# ─────────────────────────────────────────────────────────────────────────────


class ExplainOutput(BaseModel):
    """Structured output from the explainability LLM call."""
    reason_for_score: str = Field(
        default="",
        description="2-3 sentence explanation of why this score was given",
    )
    matched_strengths: list[str] = Field(
        default_factory=list,
        description="3-5 key strengths that match the job description",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Skills or requirements the candidate lacks",
    )
    improvement_areas: list[str] = Field(
        default_factory=list,
        description="Areas where the candidate could improve for this role",
    )
    recommendation: str = Field(
        default="",
        description=(
            "One of: 'Strong Fit - Recommended for next round' | "
            "'Good Fit - Consider for interview' | "
            "'Partial Fit - May need additional screening' | "
            "'Weak Fit - Does not meet key requirements'"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

EXPLAIN_SYSTEM_PROMPT = """You are an expert recruiter providing professional feedback on candidate matches.
Always respond with valid JSON matching the requested format. No markdown, no explanation — only JSON."""

EXPLAIN_HUMAN_PROMPT = """Given the following match data, generate a clear, professional explanation.

Job Title: {job_title}
Candidate: {candidate_name}
Qualification Score: {score}%

Must-Have Skills Required: {must_have_skills}
Candidate Skills: {candidate_skills}
Experience Required: {exp_required}
Candidate Experience: {candidate_exp} years

Scoring Breakdown:
- Must-have match: {must_have_score:.0%}
- Experience match: {exp_score:.0%}
- Skills depth: {skills_depth:.0%}
- Project relevance: {project_score:.0%}
- Recency factor: {recency_score:.0%}

{format_instructions}

Return ONLY valid JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN FACTORY
# ─────────────────────────────────────────────────────────────────────────────


def create_explain_chain(
    model: str = "ollama/llama3",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: int = 300,
    max_retries: int = 2,
    llm: Optional[ChatLiteLLMModel] = None,
):
    """
    Create a LangChain chain for explainability generation.

    The chain: prompt → LLM → PydanticOutputParser
    With retry: if parsing fails, retries up to max_retries times.

    Args:
        model: LiteLLM model identifier
        temperature: LLM temperature (slightly higher for natural language)
        max_tokens: Max response tokens
        timeout: LLM timeout in seconds
        max_retries: Number of retries on parse failure
        llm: Optional pre-configured LLM instance (e.g., BifrostGatewayModel)

    Returns:
        A Runnable that accepts match data dict and returns ExplainOutput
    """
    parser = PydanticOutputParser(pydantic_object=ExplainOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLAIN_SYSTEM_PROMPT),
        ("human", EXPLAIN_HUMAN_PROMPT),
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
        f"Explain chain created: model={llm.model}, temperature={llm.temperature}, retries={max_retries}"
    )
    return chain
