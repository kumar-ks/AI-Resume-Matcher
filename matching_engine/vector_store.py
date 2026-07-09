"""
Vector Store — ChromaDB Multi-Field Embedding Storage & Search
================================================================

Stores and retrieves resume embeddings using ChromaDB for fast
semantic similarity search. Uses 3 separate collections for
multi-field embeddings to improve retrieval precision.

MULTI-FIELD EMBEDDING STRATEGY:
    - skills_embeddings: Technical skills, tools, technologies, certifications
    - experience_embeddings: Role titles, companies, domains, responsibilities
    - summary_embeddings: Career summary, key achievements, domain expertise

    At query time, the JD is embedded and queried against all 3 collections.
    Results are fused using Reciprocal Rank Fusion (RRF) for a combined ranking.

MULTI-TENANT ISOLATION:
    - Every embedding is tagged with client_id in metadata
    - Queries ALWAYS filter by client_id (NDA enforcement)
    - Embeddings from one client are NEVER returned for another client

CALLED BY:
    - scanner.py → ingest mode (stores embeddings)
    - run.py → match mode (queries similar candidates)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_STORE_PATH = Path("data/chroma")

# Three collections for multi-field embeddings
COLLECTION_SKILLS = "resume_skills"
COLLECTION_EXPERIENCE = "resume_experience"
COLLECTION_SUMMARY = "resume_summary"

# Weights for fusing results from each collection
FUSION_WEIGHTS = {
    "skills": 0.45,      # Skills matter most for matching
    "experience": 0.35,  # Role/domain alignment
    "summary": 0.20,     # General career context
}


class VectorStore:
    """
    ChromaDB-based vector store with multi-field embeddings and client isolation.

    Stores 3 embeddings per resume (skills, experience, summary) and fuses
    results at query time for better retrieval precision.

    Usage:
        vs = VectorStore(persist_path="data/chroma")
        vs.store_multi_field(file_hash="abc", client_id="C1", job_id="J1",
                            skills_text="...", experience_text="...", summary_text="...",
                            metadata={...})
        results = vs.query_similar(jd_text="...", client_id="C1", top_n=20)
    """

    def __init__(self, persist_path: str | Path = DEFAULT_VECTOR_STORE_PATH):
        """
        Initialize ChromaDB with 3 collections for multi-field embeddings.

        Args:
            persist_path: Directory where ChromaDB stores its data files
        """
        import chromadb
        import os

        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_path))

        # Create 3 collections for multi-field embeddings
        self.skills_collection = self.client.get_or_create_collection(
            name=COLLECTION_SKILLS,
            metadata={"description": "Skills, tools, technologies, certifications"},
        )
        self.experience_collection = self.client.get_or_create_collection(
            name=COLLECTION_EXPERIENCE,
            metadata={"description": "Role titles, companies, domains"},
        )
        self.summary_collection = self.client.get_or_create_collection(
            name=COLLECTION_SUMMARY,
            metadata={"description": "Career summary, achievements, expertise"},
        )

        # Also keep the legacy single collection for backward compatibility
        self.collection = self.client.get_or_create_collection(
            name="resume_embeddings",
            metadata={"description": "Legacy single-field embeddings"},
        )

        self._embedding_model = None

        total = (self.skills_collection.count() +
                 self.experience_collection.count() +
                 self.summary_collection.count())
        logger.info(
            f"VectorStore initialized: {self.persist_path} "
            f"(skills={self.skills_collection.count()}, "
            f"experience={self.experience_collection.count()}, "
            f"summary={self.summary_collection.count()}, total={total})"
        )

    @property
    def embedding_model(self):
        """Lazy-load the sentence-transformers model (with SSL fix)."""
        if self._embedding_model is None:
            import httpx
            from sentence_transformers import SentenceTransformer
            import os

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    # STORE (MULTI-FIELD)
    # ─────────────────────────────────────────────────────────────────────────

    def store_multi_field(
        self,
        file_hash: str,
        client_id: str,
        job_id: str,
        skills_text: str,
        experience_text: str,
        summary_text: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store 3 separate embeddings for a resume (skills, experience, summary).

        Args:
            file_hash: Unique identifier for this resume (MD5 of file)
            client_id: Client identifier (NDA isolation boundary)
            job_id: Job opening identifier
            skills_text: Curated text of skills/technologies/certifications
            experience_text: Role titles, companies, domains
            summary_text: Career summary, achievements, expertise
            metadata: Optional additional metadata (source_file, name, etc.)

        Raises:
            ValueError: If client_id or job_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for storing embeddings (NDA enforcement)")
        if not job_id or not job_id.strip():
            raise ValueError("job_id is required for storing embeddings")

        # Build metadata (always includes client_id and job_id)
        meta = {
            "client_id": client_id.strip(),
            "job_id": job_id.strip(),
            "file_hash": file_hash,
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is None:
                    meta[k] = ""
                else:
                    meta[k] = str(v)

        doc_id = f"{client_id.strip()}_{file_hash}"

        # Embed and store in each collection
        if skills_text.strip():
            skills_emb = self._embed([skills_text[:1500]])[0]
            self.skills_collection.upsert(
                ids=[doc_id], embeddings=[skills_emb],
                documents=[skills_text[:1500]], metadatas=[meta],
            )

        if experience_text.strip():
            exp_emb = self._embed([experience_text[:1500]])[0]
            self.experience_collection.upsert(
                ids=[doc_id], embeddings=[exp_emb],
                documents=[experience_text[:1500]], metadatas=[meta],
            )

        if summary_text.strip():
            sum_emb = self._embed([summary_text[:1500]])[0]
            self.summary_collection.upsert(
                ids=[doc_id], embeddings=[sum_emb],
                documents=[summary_text[:1500]], metadatas=[meta],
            )

        logger.debug(
            f"Stored multi-field embeddings for hash={file_hash[:8]}... "
            f"client={client_id}, job={job_id}"
        )

    def store_embedding(
        self,
        file_hash: str,
        text: str,
        client_id: str,
        job_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Legacy single-field store. Kept for backward compatibility.
        New code should use store_multi_field() instead.
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for storing embeddings (NDA enforcement)")
        if not job_id or not job_id.strip():
            raise ValueError("job_id is required for storing embeddings")

        embed_text = text[:2000] if len(text) > 2000 else text
        embedding = self._embed([embed_text])[0]

        meta = {"client_id": client_id.strip(), "job_id": job_id.strip()}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is None:
                    meta[k] = ""
                else:
                    meta[k] = str(v)

        doc_id = f"{client_id.strip()}_{file_hash}"
        self.collection.upsert(
            ids=[doc_id], embeddings=[embedding],
            documents=[embed_text], metadatas=[meta],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY (MULTI-FIELD WITH RECIPROCAL RANK FUSION)
    # ─────────────────────────────────────────────────────────────────────────

    def query_similar(self, jd_text: str, client_id: str, top_n: int = 20) -> list[dict]:
        """
        Find top-N similar resumes using multi-field query with rank fusion.

        Queries all 3 collections (skills, experience, summary) and fuses
        results using weighted Reciprocal Rank Fusion (RRF).

        Falls back to legacy single collection if multi-field collections are empty.

        NDA ENFORCEMENT: Only returns embeddings belonging to the given client_id.

        Args:
            jd_text: The job description text to match against
            client_id: Client identifier — strict isolation filter
            top_n: Number of results to return (default: 20)

        Returns:
            List of dicts with 'file_hash', 'distance', 'metadata', 'document', 'rrf_score'
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for querying embeddings (NDA enforcement)")

        # Check if multi-field collections have data
        has_multi = self.skills_collection.count() > 0

        if has_multi:
            return self._query_multi_field(jd_text, client_id, top_n)
        else:
            # Fallback to legacy single collection
            return self._query_legacy(jd_text, client_id, top_n)

    def _query_multi_field(self, jd_text: str, client_id: str, top_n: int) -> list[dict]:
        """Query all 3 collections and fuse with weighted RRF."""
        jd_embedding = self._embed([jd_text[:2000]])[0]
        query_n = min(top_n * 3, 100)  # Over-fetch for better fusion

        # Query each collection
        results_by_field = {}
        collections = {
            "skills": self.skills_collection,
            "experience": self.experience_collection,
            "summary": self.summary_collection,
        }

        for field_name, coll in collections.items():
            if coll.count() == 0:
                results_by_field[field_name] = []
                continue

            actual_n = min(query_n, coll.count())
            try:
                res = coll.query(
                    query_embeddings=[jd_embedding],
                    n_results=actual_n,
                    where={"client_id": client_id.strip()},
                )
                results_by_field[field_name] = res
            except Exception as e:
                logger.warning(f"Query failed for {field_name} collection: {e}")
                results_by_field[field_name] = []

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}  # doc_id → fused score
        doc_metadata: dict[str, dict] = {}
        doc_documents: dict[str, str] = {}
        k = 60  # RRF constant

        for field_name, res in results_by_field.items():
            if not res or not res.get("ids") or not res["ids"][0]:
                continue

            weight = FUSION_WEIGHTS.get(field_name, 0.33)

            for rank, doc_id in enumerate(res["ids"][0]):
                rrf_score = weight * (1.0 / (k + rank + 1))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

                # Store metadata from first occurrence
                if doc_id not in doc_metadata and res.get("metadatas"):
                    doc_metadata[doc_id] = res["metadatas"][0][rank]
                if doc_id not in doc_documents and res.get("documents"):
                    doc_documents[doc_id] = res["documents"][0][rank]

        # Sort by fused RRF score (higher = more relevant)
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Format results
        prefix = f"{client_id.strip()}_"
        formatted = []
        for doc_id in sorted_ids[:top_n]:
            file_hash = doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id
            formatted.append({
                "file_hash": file_hash,
                "distance": 1.0 - rrf_scores[doc_id],  # Convert to distance-like (lower = better)
                "rrf_score": rrf_scores[doc_id],
                "metadata": doc_metadata.get(doc_id, {}),
                "document": doc_documents.get(doc_id, ""),
            })

        logger.info(
            f"Multi-field query returned {len(formatted)} results for client={client_id} "
            f"(RRF fusion across {len([r for r in results_by_field.values() if r])} fields)"
        )
        return formatted

    def _query_legacy(self, jd_text: str, client_id: str, top_n: int) -> list[dict]:
        """Fallback: query the legacy single collection."""
        if self.collection.count() == 0:
            logger.warning("VectorStore is empty — no embeddings to search")
            return []

        actual_top_n = min(top_n, self.collection.count())
        jd_embedding = self._embed([jd_text[:2000]])[0]

        results = self.collection.query(
            query_embeddings=[jd_embedding],
            n_results=actual_top_n,
            where={"client_id": client_id.strip()},
        )

        formatted = []
        if results and results["ids"] and results["ids"][0]:
            prefix = f"{client_id.strip()}_"
            for i, doc_id in enumerate(results["ids"][0]):
                file_hash = doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id
                formatted.append({
                    "file_hash": file_hash,
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "document": results["documents"][0][i] if results["documents"] else "",
                })

        logger.debug(f"Legacy query returned {len(formatted)} results for client={client_id}")
        return formatted

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    def has_embedding(self, file_hash: str, client_id: str) -> bool:
        """Check if an embedding exists for the given file hash under a client."""
        try:
            doc_id = f"{client_id.strip()}_{file_hash}"
            result = self.skills_collection.get(ids=[doc_id])
            return bool(result and result["ids"])
        except Exception:
            return False

    def get_count(self, client_id: Optional[str] = None) -> int:
        """Return total number of profiles stored (based on skills collection)."""
        if client_id:
            result = self.skills_collection.get(where={"client_id": client_id.strip()})
            return len(result["ids"]) if result and result["ids"] else 0
        return self.skills_collection.count()

    def get_stored_hashes(self, client_id: str) -> set[str]:
        """Return set of all file hashes stored for a client."""
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required (NDA enforcement)")

        result = self.skills_collection.get(where={"client_id": client_id.strip()})
        if result and result["ids"]:
            prefix = f"{client_id.strip()}_"
            return {
                doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id
                for doc_id in result["ids"]
            }
        return set()

    def delete_embedding(self, file_hash: str, client_id: str) -> None:
        """Delete embeddings across all collections for a file hash."""
        doc_id = f"{client_id.strip()}_{file_hash}"
        for coll in [self.skills_collection, self.experience_collection,
                     self.summary_collection, self.collection]:
            try:
                coll.delete(ids=[doc_id])
            except Exception:
                pass
        logger.debug(f"Deleted embeddings for hash={file_hash[:8]}... client={client_id}")

    def get_status(self) -> dict:
        """Get vector store status information."""
        return {
            "persist_path": str(self.persist_path),
            "embedding_count": self.skills_collection.count(),
            "skills_count": self.skills_collection.count(),
            "experience_count": self.experience_collection.count(),
            "summary_count": self.summary_collection.count(),
            "legacy_count": self.collection.count(),
            "collection_name": "multi-field (skills + experience + summary)",
        }
