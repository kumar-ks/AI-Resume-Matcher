"""
Embedding Model Cache — Global Singleton
==========================================

Loads the sentence-transformers embedding model ONCE and provides
it to all modules that need it (vector_store.py, semantic_matching.py).

BEFORE: Model loaded fresh per request (~10 sec each time)
AFTER:  Model loaded once at first use, reused for all subsequent calls (~0 sec)

USAGE:
    from matching_engine.embedding_cache import get_embedding_model, embed_texts

    model = get_embedding_model()
    vectors = embed_texts(["text1", "text2"])
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Global singleton — loaded once, reused forever
_CACHED_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: Optional[str] = None):
    """
    Get the cached embedding model. Loads on first call, reuses thereafter.

    Args:
        model_name: Override model name (default: all-MiniLM-L6-v2)

    Returns:
        SentenceTransformer model instance
    """
    global _CACHED_MODEL, _MODEL_NAME

    if model_name and model_name != _MODEL_NAME:
        _MODEL_NAME = model_name
        _CACHED_MODEL = None  # Force reload if model changed

    if _CACHED_MODEL is None:
        import httpx
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Patch httpx to handle corporate SSL issues (Zscaler)
        _original = httpx.Client.__init__

        def _patched(self_client, *args, **kwargs):
            kwargs["verify"] = False
            _original(self_client, *args, **kwargs)

        httpx.Client.__init__ = _patched

        try:
            _CACHED_MODEL = SentenceTransformer(_MODEL_NAME)
        finally:
            httpx.Client.__init__ = _original

        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info(f"Embedding model loaded and cached: {_MODEL_NAME}")

    return _CACHED_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Compute embeddings for a list of texts using the cached model.

    Args:
        texts: List of strings to embed

    Returns:
        List of embedding vectors (each 384-dim for all-MiniLM-L6-v2)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()
