"""
Vector Store — ChromaDB Embedding Storage & Search
====================================================

Stores and retrieves resume embeddings using ChromaDB for fast
semantic similarity search. This enables instant candidate retrieval
when a new JD is provided — no need to re-embed resumes.

MULTI-TENANT ISOLATION:
    - Every embedding is tagged with client_id in metadata
    - Queries ALWAYS filter by client_id (NDA enforcement)
    - Embeddings from one client are NEVER returned for another client
    - Within the same client, embeddings can be queried across job_ids

RESPONSIBILITIES:
    - Store resume text embeddings (indexed by file_hash)
    - Query top-N similar candidates for a given JD embedding (filtered by client_id)
    - Manage the ChromaDB collection lifecycle

CALLED BY:
    - run.py → --ingest mode (stores embeddings)
    - run.py → --match mode (queries similar candidates)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_STORE_PATH = Path("data/chroma")
COLLECTION_NAME = "resume_embeddings"


class VectorStore:
    """
    ChromaDB-based vector store for resume embeddings with client isolation.

    STRICT RULE: All query operations require a client_id parameter.
    Embeddings from one client are NEVER returned for another client.

    Usage:
        vs = VectorStore(persist_path="data/chroma")
        vs.store_embedding(file_hash="abc123", text="...", client_id="C1", job_id="J1", metadata={})
        results = vs.query_similar(jd_text="...", client_id="C1", top_n=20)
    """

    def __init__(self, persist_path: str | Path = DEFAULT_VECTOR_STORE_PATH):
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_path: Directory where ChromaDB stores its data files
        """
        import chromadb

        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.persist_path))

        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Get or create the collection
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
        client_id: str,
        job_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a resume's text embedding in ChromaDB with client isolation metadata.

        Args:
            file_hash: Unique identifier for this resume (MD5 of file)
            text: The resume text to embed
            client_id: Client identifier (NDA isolation boundary)
            job_id: Job opening identifier
            metadata: Optional additional metadata dict (source_file, name, etc.)

        Raises:
            ValueError: If client_id or job_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for storing embeddings (NDA enforcement)")
        if not job_id or not job_id.strip():
            raise ValueError("job_id is required for storing embeddings")

        # Build representative text and truncate
        embed_text = text[:2000] if len(text) > 2000 else text

        # Compute embedding ourselves
        embedding = self._embed([embed_text])[0]

        # Prepare metadata — always include client_id and job_id
        meta = {
            "client_id": client_id.strip(),
            "job_id": job_id.strip(),
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is None:
                    meta[k] = ""
                else:
                    meta[k] = str(v)

        # Use composite ID: client_id + file_hash to allow same file under different clients
        doc_id = f"{client_id.strip()}_{file_hash}"

        # Upsert with pre-computed embedding
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[embed_text],
            metadatas=[meta],
        )

        logger.debug(
            f"Stored embedding for hash={file_hash[:8]}... "
            f"client={client_id}, job={job_id} ({len(embed_text)} chars)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY (ALWAYS FILTERED BY CLIENT_ID — NDA ENFORCEMENT)
    # ─────────────────────────────────────────────────────────────────────────

    def query_similar(self, jd_text: str, client_id: str, top_n: int = 20) -> list[dict]:
        """
        Find the top-N most similar resumes for a given JD text, scoped to a client.

        NDA ENFORCEMENT: Only returns embeddings belonging to the given client_id.
        Embeddings from other clients are NEVER returned.

        Args:
            jd_text: The job description text to match against
            client_id: Client identifier — strict isolation filter
            top_n: Number of results to return (default: 20)

        Returns:
            List of dicts with:
                - 'file_hash': The resume's unique hash
                - 'distance': Similarity distance (lower = more similar)
                - 'metadata': Stored metadata dict
                - 'document': The stored resume text snippet

        Raises:
            ValueError: If client_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for querying embeddings (NDA enforcement)")

        if self.collection.count() == 0:
            logger.warning("VectorStore is empty — no embeddings to search")
            return []

        # Limit top_n to collection size
        actual_top_n = min(top_n, self.collection.count())

        # Compute JD embedding
        jd_embedding = self._embed([jd_text[:2000]])[0]

        # Query ChromaDB with client_id filter (NDA ENFORCEMENT)
        results = self.collection.query(
            query_embeddings=[jd_embedding],
            n_results=actual_top_n,
            where={"client_id": client_id.strip()},
        )

        # Format results
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Extract original file_hash from composite ID (client_id_filehash)
                prefix = f"{client_id.strip()}_"
                file_hash = doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id

                formatted.append({
                    "file_hash": file_hash,
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "document": results["documents"][0][i] if results["documents"] else "",
                })

        logger.debug(
            f"Query returned {len(formatted)} results for client={client_id} "
            f"(requested top {top_n})"
        )
        return formatted

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    def has_embedding(self, file_hash: str, client_id: str) -> bool:
        """Check if an embedding exists for the given file hash under a client."""
        try:
            doc_id = f"{client_id.strip()}_{file_hash}"
            result = self.collection.get(ids=[doc_id])
            return bool(result and result["ids"])
        except Exception:
            return False

    def get_count(self, client_id: Optional[str] = None) -> int:
        """
        Return total number of embeddings stored.

        Args:
            client_id: If provided, count only for this client (slower, uses where filter).
                       If None, returns total count across all clients.
        """
        if client_id:
            result = self.collection.get(where={"client_id": client_id.strip()})
            return len(result["ids"]) if result and result["ids"] else 0
        return self.collection.count()

    def get_stored_hashes(self, client_id: str) -> set[str]:
        """
        Return set of all file hashes stored in the vector store for a client.

        Args:
            client_id: Client to get hashes for

        Raises:
            ValueError: If client_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required (NDA enforcement)")

        result = self.collection.get(where={"client_id": client_id.strip()})
        if result and result["ids"]:
            prefix = f"{client_id.strip()}_"
            return {
                doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id
                for doc_id in result["ids"]
            }
        return set()

    def delete_embedding(self, file_hash: str, client_id: str) -> None:
        """Delete an embedding by file hash for a specific client."""
        try:
            doc_id = f"{client_id.strip()}_{file_hash}"
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted embedding for hash={file_hash[:8]}... client={client_id}")
        except Exception as e:
            logger.warning(f"Failed to delete embedding {file_hash[:8]}: {e}")

    def get_status(self) -> dict:
        """Get vector store status information."""
        return {
            "persist_path": str(self.persist_path),
            "embedding_count": self.get_count(),
            "collection_name": COLLECTION_NAME,
        }
