"""
LLM Factory — Centralized Model Selection
============================================

Returns the appropriate LLM instance based on configuration:
  - If Bi-Frost is enabled → BifrostGatewayModel (routes through enterprise gateway)
  - Otherwise → ChatLiteLLMModel (direct litellm calls)

PII ROUTING RULES:
    When Bi-Frost is enabled, not all stages should go through the gateway.
    Resume text contains PII (names, emails, phones) that may not be permitted
    through external gateways depending on compliance rules.

    Default routing (configurable via BIFROST_PII_STAGES env var):
      - JD extraction (Stage 1):     → GATEWAY  (no PII in job descriptions)
      - Resume extraction (Stage 2): → LOCAL    (PII in resumes)
      - Explainability (Stage 5):    → GATEWAY  (uses anonymized scoring data)

    Override: Set BIFROST_PII_STAGES=all to route everything through gateway,
    or BIFROST_PII_STAGES=none to keep everything local.

USAGE:
    from matching_engine.chains.factory import get_llm_for_stage

    # Returns the right LLM based on stage + config
    llm = get_llm_for_stage("jd", model="ollama/llama3", temperature=0.1)
    llm = get_llm_for_stage("resume", model="ollama/llama3", temperature=0.1)
    llm = get_llm_for_stage("explain", model="ollama/llama3", temperature=0.3)
"""

import logging
import os
from typing import Optional

from matching_engine.chains.gateway import BifrostGatewayModel, create_bifrost_model
from matching_engine.chains.llm import ChatLiteLLMModel

logger = logging.getLogger(__name__)

# Stages that are allowed to route through Bi-Frost by default
# (stages with PII are kept local unless explicitly overridden)
_DEFAULT_GATEWAY_STAGES = {"jd", "explain"}
_LOCAL_ONLY_STAGES = {"resume"}


def _get_gateway_stages() -> set[str]:
    """
    Determine which stages are allowed to route through Bi-Frost.

    Controlled by BIFROST_PII_STAGES env var:
      - "all"  → all stages go through gateway
      - "none" → all stages stay local
      - not set → default (jd + explain through gateway, resume stays local)
    """
    pii_config = os.environ.get("BIFROST_PII_STAGES", "").lower().strip()

    if pii_config == "all":
        return {"jd", "resume", "explain"}
    elif pii_config == "none":
        return set()
    else:
        return _DEFAULT_GATEWAY_STAGES


def is_bifrost_enabled() -> bool:
    """Check if Bi-Frost gateway is configured and enabled."""
    return bool(os.environ.get("BIFROST_BASE_URL"))


def get_llm_for_stage(
    stage: str,
    model: str = "ollama/llama3",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 300,
) -> ChatLiteLLMModel:
    """
    Get the appropriate LLM instance for a given pipeline stage.

    Applies PII routing rules:
      - If Bi-Frost is configured AND the stage is permitted → BifrostGatewayModel
      - Otherwise → standard ChatLiteLLMModel

    Args:
        stage: Pipeline stage identifier ("jd", "resume", "explain")
        model: LiteLLM model identifier (used for local routing)
        temperature: LLM temperature
        max_tokens: Max response tokens
        timeout: Request timeout in seconds

    Returns:
        ChatLiteLLMModel (standard) or BifrostGatewayModel (gateway-routed)
    """
    # Check if Bi-Frost is configured
    if not is_bifrost_enabled():
        return ChatLiteLLMModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # Check if this stage is allowed to go through the gateway
    gateway_stages = _get_gateway_stages()

    if stage in gateway_stages:
        bifrost = create_bifrost_model(
            model=None,  # Uses BIFROST_DEFAULT_MODEL or gateway's default
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if bifrost:
            logger.debug(f"Stage '{stage}' routed through Bi-Frost gateway")
            return bifrost

    # Fallback to standard local model
    logger.debug(f"Stage '{stage}' using local model: {model}")
    return ChatLiteLLMModel(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
