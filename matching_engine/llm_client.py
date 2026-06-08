"""
LLM Client with Failover and Token Validation
===============================================

Provides a resilient LLM client that:
1. Checks token/API availability before starting the pipeline
2. Automatically fails over from primary model to fallback model
3. Logs which model is being used for transparency

CALL HIERARCHY:
    run.py → validate_llm_access()         # Pre-flight check before pipeline
    run.py → LLMClient(primary, failover)  # Create client
    pipeline stages → llm_client.completion()  # All LLM calls go through here

FAILOVER LOGIC:
    1. Try primary model (e.g., anthropic/claude-3-sonnet)
    2. If primary fails (auth error, rate limit, network):
       → Log warning
       → Switch to failover model (e.g., ollama/llama3)
       → All subsequent calls use failover (no ping-pong)
    3. If failover also fails → raise exception (pipeline handles gracefully)

TOKEN VALIDATION:
    Before starting the pipeline, we make a lightweight test call to verify:
    - API key is valid (for cloud models)
    - Model is available and responsive
    - Estimated token budget is sufficient
"""

import logging
import os
from typing import Optional

import litellm

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Resilient LLM client with automatic failover.

    Usage:
        client = LLMClient(primary="anthropic/claude-3-sonnet", failover="ollama/llama3")
        response = await client.completion(messages=[...], temperature=0.1)
    """

    def __init__(self, primary_model: str, failover_model: Optional[str] = None):
        """
        Initialize LLM client with primary and optional failover model.

        Args:
            primary_model: Primary LLM model identifier (tried first)
            failover_model: Fallback model if primary fails (optional)
        """
        self.primary_model = primary_model
        self.failover_model = failover_model
        self._active_model = primary_model  # Currently active model
        self._failed_over = False  # Track if we've already failed over

        logger.info(
            f"LLMClient initialized: primary={primary_model}, "
            f"failover={failover_model or 'none'}"
        )

    @property
    def active_model(self) -> str:
        """Returns the currently active model (primary or failover)."""
        return self._active_model

    @property
    def is_failed_over(self) -> bool:
        """Returns True if currently using the failover model."""
        return self._failed_over

    async def completion(self, messages: list, temperature: float = 0.1,
                         max_tokens: int = 4096) -> object:
        """
        Make an LLM completion call with automatic failover.

        Called by: jd_understanding, resume_understanding, explainability

        Flow:
            1. Try active model (primary initially)
            2. If fails and failover available → switch to failover, retry
            3. If fails and no failover → raise exception

        Args:
            messages: Chat messages list [{"role": ..., "content": ...}]
            temperature: LLM temperature
            max_tokens: Max response tokens

        Returns:
            LiteLLM completion response object

        Raises:
            Exception: If both primary and failover fail
        """
        try:
            # Try the currently active model
            response = await litellm.acompletion(
                model=self._active_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response

        except Exception as primary_error:
            # If already on failover, or no failover configured, raise
            if self._failed_over or not self.failover_model:
                logger.error(
                    f"LLM call failed on {self._active_model}: {primary_error}"
                )
                raise

            # ── Failover: Switch to backup model ──────────────────────────────
            logger.warning(
                f"Primary model '{self.primary_model}' failed: {type(primary_error).__name__}: "
                f"{str(primary_error)[:100]}. Failing over to '{self.failover_model}'"
            )
            self._active_model = self.failover_model
            self._failed_over = True

            # Retry with failover model
            try:
                response = await litellm.acompletion(
                    model=self._active_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                logger.info(f"Failover successful — using {self._active_model}")
                return response

            except Exception as failover_error:
                logger.error(
                    f"Failover model '{self.failover_model}' also failed: {failover_error}"
                )
                raise


async def validate_llm_access(primary_model: str, failover_model: Optional[str] = None) -> dict:
    """
    Pre-flight check: Validate LLM access before starting the pipeline.

    Called by: run.py → run_matching() before pipeline execution.

    Checks:
        1. API key is present (for cloud models)
        2. Makes a lightweight test call to verify connectivity
        3. Reports which model is available

    Args:
        primary_model: Primary model identifier
        failover_model: Optional failover model identifier

    Returns:
        dict with:
            - "model": The model that will be used (primary or failover)
            - "status": "primary" | "failover" | "failed"
            - "message": Human-readable status message
    """
    logger.info(f"Validating LLM access: primary={primary_model}, failover={failover_model}")

    # ── Step 1: Check API key availability (for cloud models) ─────────────────
    key_status = _check_api_key(primary_model)
    if key_status:
        logger.debug(f"API key check for primary: {key_status}")

    # ── Step 2: Test primary model with a lightweight call ────────────────────
    test_messages = [{"role": "user", "content": "Reply with only the word: OK"}]

    primary_available = await _test_model(primary_model, test_messages)
    if primary_available:
        logger.info(f"✓ Primary model '{primary_model}' is available")
        return {
            "model": primary_model,
            "status": "primary",
            "message": f"Primary model '{primary_model}' is available and responding.",
        }

    logger.warning(f"✗ Primary model '{primary_model}' is not available")

    # ── Step 3: Test failover model ───────────────────────────────────────────
    if failover_model:
        failover_available = await _test_model(failover_model, test_messages)
        if failover_available:
            logger.info(f"✓ Failover model '{failover_model}' is available")
            return {
                "model": failover_model,
                "status": "failover",
                "message": (
                    f"Primary '{primary_model}' unavailable. "
                    f"Using failover '{failover_model}'."
                ),
            }

        logger.error(f"✗ Failover model '{failover_model}' also unavailable")

    # ── Both failed ───────────────────────────────────────────────────────────
    return {
        "model": None,
        "status": "failed",
        "message": (
            f"Both primary '{primary_model}' and failover '{failover_model or 'none'}' "
            f"are unavailable. Check API keys and model availability."
        ),
    }


async def _test_model(model: str, test_messages: list) -> bool:
    """
    Test if a model is available. Uses different strategies per provider:
    - Ollama: HTTP ping to server (fast, doesn't require model loading)
    - Cloud APIs: Lightweight completion call with timeout
    """
    import httpx

    # For Ollama models, just check if the server is reachable
    if model.startswith("ollama/"):
        try:
            async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    # Check if the specific model is available
                    data = resp.json()
                    model_name = model.replace("ollama/", "")
                    available_models = [m.get("name", "") for m in data.get("models", [])]
                    if any(model_name in m for m in available_models):
                        logger.debug(f"Ollama server reachable, model '{model_name}' available")
                        return True
                    else:
                        logger.warning(f"Ollama server reachable but model '{model_name}' not found. Available: {available_models}")
                        return False
        except Exception as e:
            logger.debug(f"Ollama server not reachable: {e}")
            return False

    # For cloud APIs, make a lightweight test call
    try:
        response = await litellm.acompletion(
            model=model,
            messages=test_messages,
            temperature=0.0,
            max_tokens=5,
            timeout=15,
        )
        return True
    except Exception as e:
        logger.debug(f"Model '{model}' test call failed: {type(e).__name__}: {str(e)[:100]}")
        return False


def _check_api_key(model: str) -> Optional[str]:
    """
    Check if the required API key is set for a given model provider.

    Returns:
        Status message, or None if no key is needed (e.g., Ollama).
    """
    if model.startswith("ollama/"):
        return None  # Ollama doesn't need an API key

    if model.startswith("anthropic/"):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return "WARNING: ANTHROPIC_API_KEY not set"
        return f"ANTHROPIC_API_KEY is set ({key[:8]}...)"

    if model.startswith("gpt") or model.startswith("openai/"):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return "WARNING: OPENAI_API_KEY not set"
        return f"OPENAI_API_KEY is set ({key[:8]}...)"

    if model.startswith("bedrock/"):
        key = os.environ.get("AWS_ACCESS_KEY_ID")
        if not key:
            return "WARNING: AWS_ACCESS_KEY_ID not set"
        return "AWS credentials are set"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
# Estimates total token usage before starting the pipeline.
# This helps users understand cost and whether they have sufficient quota.
# ─────────────────────────────────────────────────────────────────────────────

# Average tokens per character (rough estimate, varies by language)
CHARS_PER_TOKEN = 4  # English text averages ~4 characters per token

# Token overhead per LLM call (system prompt + JSON schema in the prompt)
PROMPT_OVERHEAD_TOKENS = 500

# Estimated output tokens per stage
STAGE_OUTPUT_TOKENS = {
    "jd_understanding": 1500,       # Stage 1: JD extraction response
    "resume_understanding": 2500,   # Stage 2: Resume extraction response
    "explainability": 800,          # Stage 5: Explanation response
}

# Model context window sizes (approximate)
MODEL_CONTEXT_WINDOWS = {
    "ollama/llama3": 8192,
    "ollama/llama3:70b": 8192,
    "ollama/mistral": 32768,
    "ollama/qwen2": 32768,
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    "anthropic/claude-3-sonnet-20240229": 200000,
    "anthropic/claude-3-opus-20240229": 200000,
}

# Approximate cost per 1K tokens (input + output combined, USD)
MODEL_COST_PER_1K = {
    "ollama/llama3": 0.0,           # Free (local)
    "ollama/mistral": 0.0,          # Free (local)
    "gpt-4": 0.06,                  # $0.03 input + $0.06 output per 1K
    "gpt-4o": 0.015,                # $0.005 input + $0.015 output per 1K
    "gpt-3.5-turbo": 0.002,         # $0.0005 input + $0.0015 output per 1K
    "anthropic/claude-3-sonnet-20240229": 0.015,  # $0.003 input + $0.015 output
    "anthropic/claude-3-opus-20240229": 0.075,    # $0.015 input + $0.075 output
}


def estimate_token_usage(
    jd_text: str,
    resume_texts: list[str],
    model: str,
    explain_top_n: Optional[int] = None,
) -> dict:
    """
    Estimate total token usage for the pipeline run BEFORE execution.

    Called by: run.py → run_matching() after file loading, before pipeline start.

    Calculates:
        - Input tokens: JD + all resumes + prompt overhead
        - Output tokens: Expected responses from all LLM calls
        - Total tokens: Sum of input + output
        - Estimated cost: For cloud models (Ollama is free)
        - Context window check: Warns if any single input exceeds model's limit

    Args:
        jd_text: Raw JD text
        resume_texts: List of raw resume texts
        model: Model identifier (for cost and context window lookup)
        explain_top_n: How many candidates get LLM explanations (None = all)

    Returns:
        dict with:
            - total_input_tokens: Estimated input tokens across all calls
            - total_output_tokens: Estimated output tokens
            - total_tokens: Grand total
            - estimated_cost_usd: Estimated cost in USD (0.0 for local models)
            - num_llm_calls: Total number of LLM API calls
            - context_window: Model's context window size
            - warnings: List of warning strings (e.g., "Resume X exceeds context window")
            - sufficient: Boolean — whether the pipeline can proceed
    """
    logger.info("Estimating token usage for pipeline run")

    num_resumes = len(resume_texts)
    explain_count = explain_top_n if explain_top_n else num_resumes

    # ── Estimate input tokens ─────────────────────────────────────────────────
    jd_tokens = len(jd_text) // CHARS_PER_TOKEN
    resume_tokens = [len(text) // CHARS_PER_TOKEN for text in resume_texts]

    # Stage 1: 1 call (JD text + prompt overhead)
    stage1_input = jd_tokens + PROMPT_OVERHEAD_TOKENS

    # Stage 2: N calls (each resume + prompt overhead)
    stage2_input = sum(t + PROMPT_OVERHEAD_TOKENS for t in resume_tokens)

    # Stage 5: explain_count calls (scoring data + prompt overhead, ~300 tokens input each)
    stage5_input = explain_count * (300 + PROMPT_OVERHEAD_TOKENS)

    total_input_tokens = stage1_input + stage2_input + stage5_input

    # ── Estimate output tokens ────────────────────────────────────────────────
    stage1_output = STAGE_OUTPUT_TOKENS["jd_understanding"]
    stage2_output = num_resumes * STAGE_OUTPUT_TOKENS["resume_understanding"]
    stage5_output = explain_count * STAGE_OUTPUT_TOKENS["explainability"]

    total_output_tokens = stage1_output + stage2_output + stage5_output

    # ── Total ─────────────────────────────────────────────────────────────────
    total_tokens = total_input_tokens + total_output_tokens

    # ── Number of LLM calls ───────────────────────────────────────────────────
    num_llm_calls = 1 + num_resumes + explain_count  # Stage 1 + Stage 2 + Stage 5

    # ── Cost estimate ─────────────────────────────────────────────────────────
    cost_per_1k = MODEL_COST_PER_1K.get(model, 0.01)  # Default $0.01/1K if unknown
    estimated_cost = (total_tokens / 1000) * cost_per_1k

    # ── Context window check ──────────────────────────────────────────────────
    context_window = MODEL_CONTEXT_WINDOWS.get(model, 8192)  # Default 8K if unknown
    warnings = []

    # Check if JD + prompt fits in context
    if stage1_input > context_window * 0.8:
        warnings.append(f"JD text ({jd_tokens} tokens) may be too large for {model} context window ({context_window})")

    # Check each resume
    for i, rt in enumerate(resume_tokens):
        if rt + PROMPT_OVERHEAD_TOKENS > context_window * 0.8:
            warnings.append(f"Resume {i+1} ({rt} tokens) may exceed {model} context window ({context_window})")

    # Check total feasibility
    sufficient = len([w for w in warnings if "too large" in w or "exceed" in w]) == 0

    result = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 4),
        "num_llm_calls": num_llm_calls,
        "num_resumes": num_resumes,
        "context_window": context_window,
        "model": model,
        "warnings": warnings,
        "sufficient": sufficient,
    }

    logger.debug(f"Token estimation: {result}")
    return result
