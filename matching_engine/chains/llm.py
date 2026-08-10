"""
Custom LangChain Chat Model wrapping LiteLLM
=============================================

Provides a LangChain-compatible ChatModel that uses litellm under the hood.
This avoids the problematic langchain-litellm/langchain-community packages
while giving us full LangChain Runnable composition (chains, retry, fallback).

USAGE:
    from matching_engine.chains.llm import ChatLiteLLMModel

    llm = ChatLiteLLMModel(model="ollama/llama3", temperature=0.1)
    response = await llm.ainvoke([HumanMessage(content="Hello")])

FEATURES:
    - Async-first (ainvoke/agenerate)
    - Supports all litellm model identifiers (ollama/, openai/, anthropic/, bedrock/)
    - Token usage tracking in response metadata
    - Timeout configuration
    - Compatible with LangChain's .with_retry(), .with_fallbacks(), pipe operator
"""

import logging
from typing import Any, Iterator, List, Optional

import litellm
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


def _convert_messages(messages: List[BaseMessage]) -> List[dict[str, str]]:
    """Convert LangChain messages to litellm message format."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
        else:
            # Default to user role for unknown message types
            result.append({"role": "user", "content": msg.content})
    return result


class ChatLiteLLMModel(BaseChatModel):
    """
    LangChain ChatModel backed by litellm.

    Supports all providers that litellm supports: Ollama, OpenAI, Anthropic,
    AWS Bedrock, Azure, Google Vertex, etc.

    This is a thin wrapper that delegates to litellm.acompletion() for async
    and litellm.completion() for sync calls.
    """

    model: str = "ollama/llama3"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 300  # seconds
    model_kwargs: dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "litellm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation via litellm.completion()."""
        litellm_messages = _convert_messages(messages)

        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
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
            llm_output={"token_usage": usage, "model": self.model},
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation via litellm.acompletion()."""
        litellm_messages = _convert_messages(messages)

        response = await litellm.acompletion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
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
            llm_output={"token_usage": usage, "model": self.model},
        )
