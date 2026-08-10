"""
LangFuse Observability Module
==============================

Centralized observability for the AI Resume Matcher pipeline.
Provides trace/span/generation helpers that wrap LangFuse SDK v4 calls.

DESIGN:
    - Graceful degradation: If LangFuse is not configured (missing env vars),
      all helper functions become no-ops. The pipeline runs normally without tracing.
    - Trace hierarchy: API request → Pipeline run → Stage span → LLM generation
    - Multi-tenant tagging: Every trace is tagged with client_id + job_id
    - Score tracking: qualification_percentage is recorded as a LangFuse score

ENVIRONMENT VARIABLES:
    LANGFUSE_PUBLIC_KEY   — LangFuse project public key
    LANGFUSE_SECRET_KEY   — LangFuse project secret key
    LANGFUSE_HOST         — LangFuse server URL (default: https://cloud.langfuse.com)
    LANGFUSE_ENABLED      — Set to "false" to disable tracing (default: true)

LANGFUSE SDK v4 API:
    - client.start_observation(name=..., as_type="span") → creates a root span (trace)
    - span.start_observation(name=..., as_type="span") → creates child span
    - span.start_observation(name=..., as_type="generation", model=...) → LLM call
    - span.end(output=...) → ends the span
    - span.score_trace(name=..., value=...) → records a score on the trace
    - span.update(metadata=...) → updates metadata

USAGE:
    from matching_engine.observability import get_langfuse, create_trace, create_span

    # Top-level trace (created per API request or CLI run)
    trace = create_trace(name="match-pipeline", client_id="ACME", job_id="JOB-001")

    # Stage span (child of trace)
    span = create_span(trace, name="jd-understanding")

    # LLM generation (child of span)
    generation = create_generation(
        parent=span,
        name="extract-jd",
        model="ollama/llama3",
        input_data={"prompt": "..."},
    )
    # ... call LLM ...
    end_generation(generation, output="...", usage={"total_tokens": 1500})

    end_span(span)
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LANGFUSE CLIENT SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_langfuse_client = None
_langfuse_enabled: Optional[bool] = None


def _is_enabled() -> bool:
    """Check if LangFuse tracing is enabled via env vars."""
    global _langfuse_enabled
    if _langfuse_enabled is not None:
        return _langfuse_enabled

    enabled_flag = os.environ.get("LANGFUSE_ENABLED", "true").lower()
    has_keys = bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )
    _langfuse_enabled = enabled_flag != "false" and has_keys

    if not _langfuse_enabled:
        logger.info(
            "LangFuse tracing disabled (missing LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY "
            "or LANGFUSE_ENABLED=false)"
        )
    return _langfuse_enabled


def get_langfuse():
    """
    Get or create the singleton LangFuse client.

    Returns None if LangFuse is not configured — callers should check for None.
    """
    global _langfuse_client

    if not _is_enabled():
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info(
            f"LangFuse client initialized (host={os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')})"
        )
        return _langfuse_client

    except Exception as e:
        logger.warning(f"Failed to initialize LangFuse client: {e}")
        _langfuse_enabled = False
        return None


def flush() -> None:
    """Flush any pending LangFuse events. Call on shutdown or after batch completion."""
    client = get_langfuse()
    if client:
        try:
            client.flush()
        except Exception as e:
            logger.debug(f"LangFuse flush error (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TRACE MANAGEMENT (v4 API: start_observation with as_type)
# ─────────────────────────────────────────────────────────────────────────────


def create_trace(
    name: str,
    client_id: str,
    job_id: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> Optional[Any]:
    """
    Create a top-level LangFuse span (trace root) for a pipeline run.

    In LangFuse v4, traces are the root observations. We create a root span
    that serves as the trace container for all child spans and generations.

    Args:
        name: Trace name (e.g., "ingest-pipeline", "match-pipeline")
        client_id: Client identifier (NDA tenant isolation)
        job_id: Job opening identifier
        user_id: Optional user/API key identifier
        session_id: Optional session grouping ID
        metadata: Additional metadata dict
        tags: Tags for filtering in LangFuse UI

    Returns:
        LangFuse Span object (root observation), or None if tracing is disabled.
    """
    client = get_langfuse()
    if not client:
        return None

    try:
        trace_metadata = {
            "client_id": client_id,
            "job_id": job_id,
            **(metadata or {}),
        }
        if tags:
            trace_metadata["tags"] = tags

        span = client.start_observation(
            name=name,
            as_type="span",
            metadata=trace_metadata,
        )
        logger.debug(f"LangFuse trace created: {name} (client={client_id}, job={job_id})")
        return span

    except Exception as e:
        logger.warning(f"Failed to create LangFuse trace: {e}")
        return None


def create_span(
    parent: Optional[Any],
    name: str,
    metadata: Optional[dict[str, Any]] = None,
    input_data: Optional[Any] = None,
) -> Optional[Any]:
    """
    Create a span (sub-operation) within a parent span.

    Args:
        parent: Parent span object (from create_trace or create_span)
        name: Span name (e.g., "stage-1-jd-understanding", "resume-3-extraction")
        metadata: Additional metadata
        input_data: Input data for the span

    Returns:
        LangFuse Span object, or None if tracing is disabled or parent is None.
    """
    if parent is None:
        return None

    try:
        span = parent.start_observation(
            name=name,
            as_type="span",
            input=input_data,
            metadata=metadata,
        )
        logger.debug(f"LangFuse span created: {name}")
        return span

    except Exception as e:
        logger.warning(f"Failed to create LangFuse span: {e}")
        return None


def end_span(
    span: Optional[Any],
    output: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
) -> None:
    """
    End a span with optional output and metadata.

    Args:
        span: Span object to end
        output: Output data from the span
        metadata: Updated metadata
        level: Log level ("DEBUG", "DEFAULT", "WARNING", "ERROR")
        status_message: Status message for error reporting
    """
    if span is None:
        return

    try:
        # Update metadata/level before ending if needed
        update_kwargs: dict[str, Any] = {}
        if metadata is not None:
            update_kwargs["metadata"] = metadata
        if level is not None:
            update_kwargs["level"] = level
        if status_message is not None:
            update_kwargs["status_message"] = status_message

        if update_kwargs:
            span.update(**update_kwargs)

        # End the span
        end_kwargs: dict[str, Any] = {}
        if output is not None:
            end_kwargs["output"] = output
        span.end(**end_kwargs)

    except Exception as e:
        logger.debug(f"Failed to end LangFuse span: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM GENERATION TRACKING
# ─────────────────────────────────────────────────────────────────────────────


def create_generation(
    parent: Optional[Any],
    name: str,
    model: str,
    input_data: Optional[Any] = None,
    model_parameters: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Create a generation event for an LLM call.

    In LangFuse v4, generations are observations with as_type="generation".

    Args:
        parent: Parent span or trace
        name: Generation name (e.g., "extract-jd-requirements")
        model: Model identifier (e.g., "ollama/llama3")
        input_data: Prompt/messages sent to the LLM
        model_parameters: Parameters like temperature, max_tokens
        metadata: Additional metadata

    Returns:
        LangFuse Generation object, or None if tracing is disabled.
    """
    if parent is None:
        return None

    try:
        generation = parent.start_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input_data,
            model_parameters=model_parameters,
            metadata=metadata,
        )
        logger.debug(f"LangFuse generation created: {name} (model={model})")
        return generation

    except Exception as e:
        logger.warning(f"Failed to create LangFuse generation: {e}")
        return None


def end_generation(
    generation: Optional[Any],
    output: Optional[Any] = None,
    usage: Optional[dict[str, int]] = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
) -> None:
    """
    End a generation with output and token usage.

    Args:
        generation: Generation object to end
        output: LLM response content
        usage: Token usage dict with keys: input_tokens, output_tokens, total_tokens
        level: Log level ("DEFAULT", "WARNING", "ERROR")
        status_message: Error message if generation failed
    """
    if generation is None:
        return

    try:
        # Update level/status if needed
        update_kwargs: dict[str, Any] = {}
        if level is not None:
            update_kwargs["level"] = level
        if status_message is not None:
            update_kwargs["status_message"] = status_message

        if update_kwargs:
            generation.update(**update_kwargs)

        # End with output and usage
        end_kwargs: dict[str, Any] = {}
        if output is not None:
            end_kwargs["output"] = output
        if usage is not None:
            # LangFuse v4 uses usage_details
            end_kwargs["usage_details"] = {
                "input": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                "output": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                "total": usage.get("total_tokens", 0),
            }

        generation.end(**end_kwargs)

    except Exception as e:
        logger.debug(f"Failed to end LangFuse generation: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SCORE TRACKING
# ─────────────────────────────────────────────────────────────────────────────


def score_trace(
    trace: Optional[Any],
    name: str,
    value: float,
    comment: Optional[str] = None,
) -> None:
    """
    Record a score on a trace (e.g., qualification_percentage).

    Args:
        trace: Trace/span object to score
        name: Score name (e.g., "qualification_percentage", "semantic_similarity")
        value: Score value (0.0 - 1.0 for normalized, or raw percentage)
        comment: Optional comment explaining the score
    """
    if trace is None:
        return

    try:
        kwargs: dict[str, Any] = {"name": name, "value": value}
        if comment:
            kwargs["comment"] = comment
        trace.score_trace(**kwargs)
        logger.debug(f"LangFuse score recorded: {name}={value}")

    except Exception as e:
        logger.debug(f"Failed to record LangFuse score: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Extract token usage from litellm response
# ─────────────────────────────────────────────────────────────────────────────


def extract_usage_from_response(response: Any) -> Optional[dict[str, int]]:
    """
    Extract token usage from a litellm response object.

    Args:
        response: litellm completion response

    Returns:
        dict with input_tokens, output_tokens, total_tokens — or None if not available
    """
    try:
        usage = response.usage
        if usage:
            return {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
    except (AttributeError, TypeError):
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT MANAGER for span lifecycle
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def traced_span(
    parent: Optional[Any],
    name: str,
    metadata: Optional[dict[str, Any]] = None,
    input_data: Optional[Any] = None,
):
    """
    Context manager that creates a span on enter and ends it on exit.

    Usage:
        with traced_span(trace, "stage-1-jd") as span:
            # ... do work ...
            # span is auto-ended on exit (even on exception)

    On exception, the span is ended with level="ERROR" and the exception message.
    """
    span = create_span(parent, name, metadata=metadata, input_data=input_data)
    try:
        yield span
    except Exception as e:
        end_span(span, level="ERROR", status_message=str(e))
        raise
    else:
        end_span(span)
