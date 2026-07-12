"""
Basic Authentication for the AI Resume Matcher API.

Uses API key-based auth via X-API-Key header.
Keys are loaded from environment variable AI_MATCHER_API_KEYS (comma-separated).
"""

import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_keys() -> set[str]:
    """Load valid API keys from environment."""
    keys_str = os.environ.get("AI_MATCHER_API_KEYS", "")
    if not keys_str:
        # Default dev key if none set
        return {"dev-key-change-me"}
    return {k.strip() for k in keys_str.split(",") if k.strip()}


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify the API key from request header."""
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    valid_keys = get_api_keys()
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key
