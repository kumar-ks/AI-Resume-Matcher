"""
Bi-Frost Gateway Model
========================

A ChatLiteLLMModel subclass that routes LLM calls through BT's GenAI Gateway
(Bi-Frost). The gateway provides:
  - Centralized auth & token management
  - Model governance & compliance controls
  - Audit trail for all LLM interactions
  - Automatic failover & load balancing across providers
  - Content filtering & guardrails

ARCHITECTURE:
    Bi-Frost exposes an OpenAI-compatible API. We route litellm calls through
    it by setting api_base to the gateway URL and passing the gateway API key.
    Additional metadata headers (project_id, cost_center, use_case) are sent
    for cost attribution and compliance tagging.

USAGE:
    from matching_engine.chains.gateway import BifrostGatewayModel

    llm = BifrostGatewayModel(
        model="anthropic/claude-3-sonnet",
        gateway_url="https://bifrost.internal.bt.com/v1",
        gateway_api_key="bf-...",
        project_id="ai-resume-matcher",
        cost_center="CC-1234",
        use_case="recruitment-automation",
    )
    response = await llm.ainvoke([HumanMessage(content="Hello")])

ENVIRONMENT VARIABLES:
    BIFROST_BASE_URL     — Gateway endpoint (e.g., https://bifrost.internal.bt.com/v1)
    BIFROST_API_KEY      — Gateway authentication key
    BIFROST_PROJECT_ID   — Project identifier for cost attribution
    BIFROST_COST_CENTER  — Cost center code for billing
    BIFROST_USE_CASE     — Use case classification tag
    BIFROST_DEFAULT_MODEL — Default model to use via gateway (overrides config.yaml model)
"""

import logging
import os
from typing import Any, List, Optional

import litellm
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from matching_engine.chains.llm import ChatLiteLLMModel, _convert_messages

logger = logging.getLogger(__name__)


class BifrostGatewayModel(ChatLiteLLMModel):
    """
    LangChain ChatModel that routes through BT's Bi-Frost GenAI Gateway.

    Inherits from ChatLiteLLMModel and overrides the generation methods to:
      1. Set api_base to the gateway URL
      2. Pass gateway_api_key as the auth token
      3. Include metadata headers for governance/compliance
      4. Map the model identifier through the gateway's namespace

    If the gateway is unreachable, raises an exception (no silent fallback —
    gateway routing is a compliance requirement when enabled).
    """

    # Gateway connection
    gateway_url: str = ""
    gateway_api_key: str = ""

    # Metadata for governance/compliance
    project_id: str = ""
    cost_center: str = ""
    use_case: str = ""

    @property
    def _llm_type(self) -> str:
        return "bifrost-gateway"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        params = super()._identifying_params
        params.update({
            "gateway_url": self.gateway_url,
            "project_id": self.project_id,
            "cost_center": self.cost_center,
            "use_case": self.use_case,
        })
        return params

    def _get_gateway_kwargs(self) -> dict[str, Any]:
        """Build the extra kwargs for litellm calls routed through the gateway."""
        kwargs: dict[str, Any] = {}

        # Route through the gateway endpoint
        if self.gateway_url:
            kwargs["api_base"] = self.gateway_url

        # Gateway auth key (passed as the API key to litellm)
        if self.gateway_api_key:
            kwargs["api_key"] = self.gateway_api_key

        # Metadata headers for governance — passed via litellm's extra_headers
        headers = {}
        if self.project_id:
            headers["X-Project-Id"] = self.project_id
        if self.cost_center:
            headers["X-Cost-Center"] = self.cost_center
        if self.use_case:
            headers["X-Use-Case"] = self.use_case

        if headers:
            kwargs["extra_headers"] = headers

        return kwargs

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation routed through Bi-Frost gateway."""
        litellm_messages = _convert_messages(messages)
        gateway_kwargs = self._get_gateway_kwargs()

        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            **gateway_kwargs,
            **self.model_kwargs,
            **kwargs,
        )

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }

        generation = ChatGeneration(
            message=AIMessage(content=content, response_metadata=usage),
        )
        return ChatResult(
            generations=[generation],
            llm_output={"token_usage": usage, "model": self.model, "gateway": "bifrost"},
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation routed through Bi-Frost gateway."""
        litellm_messages = _convert_messages(messages)
        gateway_kwargs = self._get_gateway_kwargs()

        response = await litellm.acompletion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            **gateway_kwargs,
            **self.model_kwargs,
            **kwargs,
        )

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }

        generation = ChatGeneration(
            message=AIMessage(content=content, response_metadata=usage),
        )
        return ChatResult(
            generations=[generation],
            llm_output={"token_usage": usage, "model": self.model, "gateway": "bifrost"},
        )


def create_bifrost_model(
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 300,
) -> Optional[BifrostGatewayModel]:
    """
    Create a BifrostGatewayModel from environment variables.

    Returns None if Bi-Frost is not configured (missing BIFROST_BASE_URL).
    This allows callers to gracefully fall back to standard ChatLiteLLMModel.

    Args:
        model: Override model identifier (defaults to BIFROST_DEFAULT_MODEL env or config)
        temperature: LLM temperature
        max_tokens: Max response tokens
        timeout: Request timeout in seconds

    Returns:
        BifrostGatewayModel if configured, None otherwise
    """
    gateway_url = os.environ.get("BIFROST_BASE_URL", "")
    gateway_api_key = os.environ.get("BIFROST_API_KEY", "")

    if not gateway_url:
        logger.debug("Bi-Frost not configured (BIFROST_BASE_URL not set)")
        return None

    resolved_model = model or os.environ.get("BIFROST_DEFAULT_MODEL", "anthropic/claude-3-sonnet")

    bifrost = BifrostGatewayModel(
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        gateway_url=gateway_url,
        gateway_api_key=gateway_api_key,
        project_id=os.environ.get("BIFROST_PROJECT_ID", ""),
        cost_center=os.environ.get("BIFROST_COST_CENTER", ""),
        use_case=os.environ.get("BIFROST_USE_CASE", "recruitment-automation"),
    )

    logger.info(
        f"Bi-Frost gateway model created: model={resolved_model}, "
        f"gateway={gateway_url}, project={bifrost.project_id}"
    )
    return bifrost
