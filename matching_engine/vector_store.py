"""
Vector Store — ChromaDB Embedding Storage & Search
====================================================

Stores and retrieves resume embeddings using ChromaDB for fast
semantic similarity search. This enables instant candidate retrieval
when a new JD is provided — no need to re-embed resumes.

RESPONSIBILITIES:
    - Store resume text embeddings (indexed by file_hash)
    - Query top-N similar candidates for a given JD embedding
    - Manage the ChromaDB collection lifecycle

CALLED BY:
    - run.py → --ingest mode (stores embeddings)
    - run.py → --match mode (queries similar candidates)

WHY ChromaDB:
    - File-based (no server needed)
    - Built-in embedding support
    - Fast approximate nearest-neighbor search
    - Handles 100K+ vectors easily
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_STORE_PATH = Path("data/chroma")
COLLECTION_NAME = "resume_embeddings"


class VectorStore:
    """
    ChromaDB-based vector store for resume embeddings.

    Usage:
        vs = VectorStore(persist_path="data/chroma")
        vs.store_embedding(file_hash="abc123", text="resume text...", metadata={...})
        results = vs.query_similar(jd_text="Job description...", top_n=20)
    """

    def __init__(self, persist_path: str | Path = DEFAULT_VECTOR_STORE_PATH):
        """
        Initialize ChromaDB with persistent storage.

        Uses a custom embedding function that bypasses SSL issues by
        using pre-computed embeddings from sentence-transformers (already
        handled by semantic_matching.py with SSL workaround).

        Args:
            persist_path: Directory where ChromaDB stores its data files
        """
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.persist_path))

        # Use sentence-transformers directly (already handles SSL via our patching)
        # We pass trust_remote_code=False and set the model to one we already have cached
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Get or create the collection — using ChromaDB without default embedding
        # We'll provide embeddings ourselves via sentence-transformers
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Resume text embeddings for semantic search"},
        )

        # Load embedding model (reuses the same model as semantic_matching.py)
        self._embedding_model = None

        logger.info(
            f"VectorStore initialized: {self.persist_path} "
            f"({self.collection.count()} embeddings)"
        )

    @property
    def embedding_model(self):
        """Lazy-load the sentence-transformers model (with SSL fix)."""
        if self._embedding_model is None:
            import httpx
            from sentence_transformers import SentenceTransformer
            import os

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Patch httpx to handle corporate SSL issues (same as semantic_matching.py)
            _original = httpx.Client.__init__
            def _patched(self_client, *args, **kwargs):
                kwargs["verify"] = False
                _original(self_client, *args, **kwargs)
            httpx.Client.__init__ = _patched

            try:
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            finally:
                httpx.Client.__init__ = _original

            os.environ["HF_HUB_OFFLINE"] = "1"
            logger.info("VectorStore embedding model loaded: all-MiniLM-L6-v2")

        return self._embedding_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings using sentence-transformers."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    # ─────────────────────────────────────────────────────────────────────────
    # STORE
    # ─────────────────────────────────────────────────────────────────────────

    def store_embedding(
        self,
        file_hash: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a resume's text embedding in ChromaDB.

        Computes embedding using sentence-transformers (same model as semantic_matching).

        Args:
            file_hash: Unique identifier for this resume (MD5 of file)
            text: The resume text to embed
            metadata: Optional metadata dict (source_file, name, etc.)
        """
        # Build representative text and truncate
        embed_text = text[:2000] if len(text) > 2000 else text

        # Compute embedding ourselves
        embedding = self._embed([embed_text])[0]

        # Prepare metadata (ChromaDB values must be str, int, float, or bool)
        meta = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is None:
                    meta[k] = ""
                else:
                    meta[k] = str(v)

        # Upsert with pre-computed embedding
        self.collection.upsert(
            ids=[file_hash],
            embeddings=[embedding],
            documents=[embed_text],
            metadatas=[meta] if meta else None,
        )

        logger.debug(f"Stored embedding for hash={file_hash[:8]}... ({len(embed_text)} chars)")

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY
    # ─────────────────────────────────────────────────────────────────────────

    def query_similar(self, jd_text: str, top_n: int = 20) -> list[dict]:
        """
        Find the top-N most similar resumes for a given JD text.

        Computes JD embedding and queries ChromaDB for nearest neighbors.

        Args:
            jd_text: The job description text to match against
            top_n: Number of results to return (default: 20)

        Returns:
            List of dicts with:
                - 'file_hash': The resume's unique hash
                - 'distance': Similarity distance (lower = more similar)
                - 'metadata': Stored metadata dict
                - 'document': The stored resume text snippet
        """
        if self.collection.count() == 0:
            logger.warning("VectorStore is empty — no embeddings to search")
            return []

        # Limit top_n to collection size
        actual_top_n = min(top_n, self.collection.count())

        # Compute JD embedding
        jd_embedding = self._embed([jd_text[:2000]])[0]

        # Query ChromaDB with pre-computed embedding
        results = self.collection.query(
            query_embeddings=[jd_embedding],
            n_results=actual_top_n,
        )

        # Format results
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, file_hash in enumerate(results["ids"][0]):
                formatted.append({
                    "file_hash": file_hash,
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "document": results["documents"][0][i] if results["documents"] else "",
                })

        logger.debug(f"Query returned {len(formatted)} results (requested top {top_n})")
        return formatted

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    def has_embedding(self, file_hash: str) -> bool:
        """Check if an embedding exists for the given file hash."""
        try:
            result = self.collection.get(ids=[file_hash])
            return bool(result and result["ids"])
        except Exception:
            return False

    def get_count(self) -> int:
        """Return total number of embeddings stored."""
        return self.collection.count()

    def get_stored_hashes(self) -> set[str]:
        """Return set of all file hashes stored in the vector store."""
        result = self.collection.get()
        return set(result["ids"]) if result and result["ids"] else set()

    def delete_embedding(self, file_hash: str) -> None:
        """Delete an embedding by file hash."""
        try:
            self.collection.delete(ids=[file_hash])
            logger.debug(f"Deleted embedding for hash={file_hash[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to delete embedding {file_hash[:8]}: {e}")

    def get_status(self) -> dict:
        """Get vector store status information."""
        return {
            "persist_path": str(self.persist_path),
            "embedding_count": self.get_count(),
            "collection_name": COLLECTION_NAME,
        }
