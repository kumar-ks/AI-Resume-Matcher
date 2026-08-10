"""
LangChain Chains for AI Resume Matcher
========================================

This package contains LangChain chain definitions for each LLM-powered
pipeline stage. Each chain encapsulates:
  - A ChatPromptTemplate with format instructions
  - A custom ChatLiteLLM model (wrapping litellm.acompletion)
  - A PydanticOutputParser for structured output validation
  - Retry logic for malformed LLM responses

MODULES:
    llm.py         — Custom ChatLiteLLM BaseChatModel wrapping litellm
    jd_chain.py    — Stage 1: JD extraction chain → JobDescription
    resume_chain.py — Stage 2: Resume extraction chain → ResumeProfile
    explain_chain.py — Stage 5: Explainability chain → ExplainabilityReport
"""

from matching_engine.chains.llm import ChatLiteLLMModel

__all__ = ["ChatLiteLLMModel"]
