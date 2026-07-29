"""
Vector Store — PostgreSQL + pgvector Embedding Storage & Search
================================================================

Stores and retrieves resume embeddings using PostgreSQL with the pgvector
extension. Replaces ChromaDB for production-grade concurrent access at scale.

MULTI-FIELD EMBEDDING STRATEGY:
    Three rows per resume (one per field type):
    - skills: Technologies, tools, certifications
    - experience: Role titles, companies, domains
    - summary: Career summary, achievements, expertise

    At query time, the JD is embedded and queried against all 3 field types.
    Results are fused using weighted Reciprocal Rank Fusion (RRF) + BM25 hybrid.

MULTI-TENANT ISOLATION:
    - Every embedding row has client_id column
    - Queries ALWAYS filter by client_id (NDA enforcement)

TABLE SCHEMA:
    resume_embeddings:
        id          SERIAL PRIMARY KEY
        client_id   TEXT NOT NULL
        job_id      TEXT NOT NULL
        file_hash   TEXT NOT NULL
        field_type  TEXT NOT NULL  ('skills', 'experience', 'summary')
        content     TEXT           (the text that was embedded)
        embedding   VECTOR(384)    (sentence-transformers output)
        metadata    JSONB          (source_file, full_name, experience_years, etc.)
        created_at  TIMESTAMPTZ
        UNIQUE(client_id, file_hash, field_type)

    Indexes:
        - HNSW index on embedding for fast ANN search
        - B-tree on client_id for partition filtering
        - GIN on content for full-text search (BM25 replacement)

CALLED BY:
    - scanner.py → ingest mode (stores embeddings)
    - run.py / api/server.py → match mode (queries similar candidates)
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "postgresql://matcher:matcher_secret@localhost:5432/resume_matcher"

# Weights for fusing results from each field type
FUSION_WEIGHTS = {
    "skills": 0.45,
    "experience": 0.35,
    "summary": 0.20,
}

# Hybrid search weights
VECTOR_WEIGHT = 0.65
BM25_WEIGHT = 0.35


class VectorStore:
    """
    PostgreSQL + pgvector based vector store with multi-field embeddings.

    Stores 3 embeddings per resume (skills, experience, summary) and fuses
    results at query time using RRF + BM25 for hybrid search.

    Usage:
        vs = VectorStore()
        vs.store_multi_field(file_hash="abc", client_id="C1", job_id="J1",
                            skills_text="...", experience_text="...", summary_text="...",
                            metadata={...})
        results = vs.query_similar(jd_text="...", client_id="C1", top_n=20)
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize PostgreSQL connection with pgvector extension.

        Args:
            database_url: PostgreSQL connection string.
                          Falls back to DATABASE_URL env var or default.
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
        self.conn = psycopg.connect(self.database_url, row_factory=dict_row)
        self._embedding_model = None
        self._setup_pgvector()
        logger.info("VectorStore initialized (PostgreSQL + pgvector)")

    def _setup_pgvector(self) -> None:
        """Create pgvector extension, table, and indexes."""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create embeddings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS resume_embeddings (
                    id SERIAL PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    field_type TEXT NOT NULL,
                    content TEXT DEFAULT '',
                    embedding VECTOR(384),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(client_id, file_hash, field_type)
                )
            """)

            # B-tree index for client filtering
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_client
                ON resume_embeddings(client_id)
            """)

            # HNSW index for fast approximate nearest neighbor search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
                ON resume_embeddings USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            # GIN index for full-text search (BM25 replacement)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_fts
                ON resume_embeddings USING gin (to_tsvector('english', content))
            """)

        self.conn.commit()
        logger.debug("pgvector extension and tables created")

    @property
    def embedding_model(self):
        """Get the cached embedding model (global singleton)."""
        if self._embedding_model is None:
            from matching_engine.embedding_cache import get_embedding_model
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings using the cached global model."""
        from matching_engine.embedding_cache import embed_texts
        return embed_texts(texts)

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
        Store 3 embeddings (skills, experience, summary) for a resume.

        Args:
            file_hash: MD5 hash of resume file
            client_id: Client identifier (NDA isolation)
            job_id: Job opening identifier
            skills_text: Curated skills/technologies text
            experience_text: Role titles, companies text
            summary_text: Career summary text
            metadata: Additional metadata (source_file, full_name, etc.)
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required (NDA enforcement)")
        if not job_id or not job_id.strip():
            raise ValueError("job_id is required")

        meta_json = json.dumps(metadata or {})
        fields = {
            "skills": skills_text,
            "experience": experience_text,
            "summary": summary_text,
        }

        for field_type, text in fields.items():
            if not text or not text.strip():
                continue

            truncated = text[:1500]
            embedding = self._embed([truncated])[0]

            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO resume_embeddings
                        (client_id, job_id, file_hash, field_type, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s::jsonb)
                    ON CONFLICT (client_id, file_hash, field_type) DO UPDATE SET
                        job_id = EXCLUDED.job_id,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW()
                """, (
                    client_id.strip(), job_id.strip(), file_hash,
                    field_type, truncated, str(embedding), meta_json,
                ))

        self.conn.commit()
        logger.debug(f"Stored multi-field embeddings: hash={file_hash[:8]}... client={client_id}")

    def store_embedding(
        self,
        file_hash: str,
        text: str,
        client_id: str,
        job_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Legacy single-field store. Stores as 'skills' field type for compat."""
        self.store_multi_field(
            file_hash=file_hash,
            client_id=client_id,
            job_id=job_id,
            skills_text=text,
            experience_text="",
            summary_text="",
            metadata=metadata,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY (HYBRID: VECTOR + FULL-TEXT SEARCH, CLIENT-SCOPED)
    # ─────────────────────────────────────────────────────────────────────────

    def query_similar(self, jd_text: str, client_id: str, top_n: int = 20) -> list[dict]:
        """
        Hybrid search: vector similarity + full-text search, fused with RRF.

        NDA ENFORCEMENT: Only returns results for the given client_id.

        Args:
            jd_text: Job description text to match against
            client_id: Client identifier (strict isolation)
            top_n: Number of results to return

        Returns:
            List of dicts with file_hash, hybrid_score, metadata
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required (NDA enforcement)")

        # Embed the JD text
        jd_embedding = self._embed([jd_text[:2000]])[0]

        # ── Vector search per field type with RRF fusion ──────────────────────
        rrf_scores: dict[str, float] = {}
        file_metadata: dict[str, dict] = {}
        k = 60  # RRF constant

        for field_type, weight in FUSION_WEIGHTS.items():
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT file_hash, metadata,
                           1 - (embedding <=> %s::vector) AS cosine_similarity
                    FROM resume_embeddings
                    WHERE client_id = %s AND field_type = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (
                    str(jd_embedding), client_id.strip(), field_type,
                    str(jd_embedding), top_n * 3,
                ))
                rows = cur.fetchall()

            for rank, row in enumerate(rows):
                fh = row["file_hash"]
                rrf_score = weight * (1.0 / (k + rank + 1))
                rrf_scores[fh] = rrf_scores.get(fh, 0.0) + rrf_score
                if fh not in file_metadata:
                    meta = row["metadata"]
                    file_metadata[fh] = meta if isinstance(meta, dict) else json.loads(meta or "{}")

        # ── Full-text search (BM25 equivalent via PostgreSQL ts_rank) ─────────
        fts_scores = self._fulltext_search(jd_text, client_id, top_n * 3)

        # ── Hybrid fusion ─────────────────────────────────────────────────────
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        max_fts = max(fts_scores.values()) if fts_scores else 1.0

        all_hashes = set(rrf_scores.keys()) | set(fts_scores.keys())
        hybrid_scores: dict[str, float] = {}

        for fh in all_hashes:
            vec_norm = (rrf_scores.get(fh, 0.0) / max_rrf) if max_rrf > 0 else 0
            fts_norm = (fts_scores.get(fh, 0.0) / max_fts) if max_fts > 0 else 0
            hybrid_scores[fh] = VECTOR_WEIGHT * vec_norm + BM25_WEIGHT * fts_norm

        # Sort and return top-N
        sorted_hashes = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)

        results = []
        for fh in sorted_hashes[:top_n]:
            results.append({
                "file_hash": fh,
                "hybrid_score": hybrid_scores[fh],
                "distance": 1.0 - hybrid_scores[fh],
                "metadata": file_metadata.get(fh, {}),
            })

        logger.info(f"Hybrid search: {len(results)} results for client={client_id}")
        return results

    def _fulltext_search(self, query_text: str, client_id: str, limit: int) -> dict[str, float]:
        """
        PostgreSQL full-text search (replaces custom BM25 implementation).

        Uses ts_rank with plainto_tsquery for relevance scoring.
        """
        # Extract meaningful terms for the query (skip very short words)
        terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b', query_text)
        if not terms:
            return {}

        # Use plainto_tsquery which handles multiple words as AND
        query_str = " ".join(terms[:30])  # Limit to 30 terms

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT file_hash, ts_rank(
                    to_tsvector('english', content),
                    plainto_tsquery('english', %s)
                ) AS rank
                FROM resume_embeddings
                WHERE client_id = %s
                  AND to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """, (query_str, client_id.strip(), query_str, limit))
            rows = cur.fetchall()

        # Aggregate scores per file_hash (max across field types)
        scores: dict[str, float] = {}
        for row in rows:
            fh = row["file_hash"]
            scores[fh] = max(scores.get(fh, 0.0), row["rank"])

        return scores

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    def has_embedding(self, file_hash: str, client_id: str) -> bool:
        """Check if embeddings exist for a file hash under a client."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM resume_embeddings WHERE client_id = %s AND file_hash = %s LIMIT 1",
                (client_id.strip(), file_hash)
            )
            return cur.fetchone() is not None

    def get_count(self, client_id: Optional[str] = None) -> int:
        """Return number of unique profiles (by file_hash) stored."""
        with self.conn.cursor() as cur:
            if client_id:
                cur.execute(
                    "SELECT COUNT(DISTINCT file_hash) as cnt FROM resume_embeddings WHERE client_id = %s",
                    (client_id.strip(),)
                )
            else:
                cur.execute("SELECT COUNT(DISTINCT file_hash) as cnt FROM resume_embeddings")
            return cur.fetchone()["cnt"]

    def get_stored_hashes(self, client_id: str) -> set[str]:
        """Return all file hashes stored for a client."""
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required (NDA enforcement)")

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT file_hash FROM resume_embeddings WHERE client_id = %s",
                (client_id.strip(),)
            )
            return {row["file_hash"] for row in cur.fetchall()}

    def delete_embedding(self, file_hash: str, client_id: str) -> None:
        """Delete all embeddings for a file hash under a client."""
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM resume_embeddings WHERE client_id = %s AND file_hash = %s",
                (client_id.strip(), file_hash)
            )
        self.conn.commit()

    def get_status(self) -> dict:
        """Get vector store status."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT file_hash) as profiles FROM resume_embeddings")
            profiles = cur.fetchone()["profiles"]
            cur.execute("SELECT COUNT(*) as total FROM resume_embeddings")
            total_rows = cur.fetchone()["total"]

        return {
            "db_type": "PostgreSQL + pgvector",
            "embedding_count": profiles,
            "total_embedding_rows": total_rows,
            "fields_per_profile": 3,
            "embedding_dim": 384,
            "index_type": "HNSW (cosine)",
        }

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
