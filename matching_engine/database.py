"""
Database Layer — PostgreSQL Profile Storage
=============================================

Manages persistent storage of extracted resume profiles in PostgreSQL.
Replaces SQLite for production-grade concurrent access at scale (5M+ profiles).

MULTI-TENANT ISOLATION (NDA ENFORCEMENT):
    - Every profile is tagged with client_id and job_id
    - ALL read queries filter by client_id
    - Resumes under one client_id can NEVER be returned for another client
    - Within the same client_id, resumes are shared across job_ids freely

SCHEMA:
    resume_profiles:
        id                      SERIAL PRIMARY KEY
        client_id               TEXT NOT NULL
        job_id                  TEXT NOT NULL
        source_file             TEXT NOT NULL
        source_path             TEXT
        file_hash               TEXT NOT NULL
        first_name              TEXT
        middle_name             TEXT
        last_name               TEXT
        email                   TEXT
        phone                   TEXT
        location                TEXT
        career_summary          TEXT
        skills                  JSONB
        total_experience_years  REAL
        work_experiences        JSONB
        projects                JSONB
        education               JSONB
        certifications          JSONB
        domain_expertise        JSONB
        raw_text                TEXT
        extracted_at            TIMESTAMPTZ
        UNIQUE(client_id, file_hash)

CONNECTION:
    Uses psycopg (sync driver) for both CLI and API modes.
    Connection string from DATABASE_URL env var or config.yaml.

CALLED BY:
    - scanner.py → ingest mode (writes profiles)
    - run.py / api/server.py → match mode (reads profiles)
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from matching_engine.models import (
    Project,
    ResumeProfile,
    WorkExperience,
)

logger = logging.getLogger(__name__)

# Default connection string (override via DATABASE_URL env var or config)
DEFAULT_DATABASE_URL = "postgresql://matcher:matcher_secret@localhost:5432/resume_matcher"


class ProfileDatabase:
    """
    PostgreSQL-based storage for extracted resume profiles with client isolation.

    STRICT RULE: All read operations require a client_id parameter.
    Profiles from one client are NEVER visible to another client.

    Usage:
        db = ProfileDatabase()
        db.store_profile(profile, source_file, source_path, file_hash, client_id, job_id)
        profiles = db.get_all_profiles(client_id="C1")
    """

    def __init__(self, database_url: Optional[str] = None, **kwargs):
        """
        Initialize PostgreSQL connection and ensure tables exist.

        Args:
            database_url: PostgreSQL connection string.
                          Falls back to DATABASE_URL env var or default.
            **kwargs: Ignored (for backward compat with old SQLite args like db_path)
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
        self.conn = psycopg.connect(self.database_url, row_factory=dict_row)
        self._create_tables()
        logger.info("ProfileDatabase initialized (PostgreSQL)")

    def _create_tables(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS resume_profiles (
                    id SERIAL PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    source_path TEXT NOT NULL DEFAULT '',
                    file_hash TEXT NOT NULL,
                    first_name TEXT DEFAULT '',
                    middle_name TEXT,
                    last_name TEXT DEFAULT '',
                    email TEXT,
                    phone TEXT,
                    location TEXT,
                    career_summary TEXT DEFAULT '',
                    skills JSONB DEFAULT '[]',
                    total_experience_years REAL,
                    work_experiences JSONB DEFAULT '[]',
                    projects JSONB DEFAULT '[]',
                    education JSONB DEFAULT '[]',
                    certifications JSONB DEFAULT '[]',
                    domain_expertise JSONB DEFAULT '[]',
                    raw_text TEXT DEFAULT '',
                    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(client_id, file_hash)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_client_id
                ON resume_profiles(client_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_client_job
                ON resume_profiles(client_id, job_id)
            """)
        self.conn.commit()
        logger.debug("PostgreSQL tables verified/created")

    # ─────────────────────────────────────────────────────────────────────────
    # WRITE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def store_profile(
        self,
        profile: ResumeProfile,
        source_file: str,
        source_path: str,
        file_hash: str,
        client_id: str,
        job_id: str,
    ) -> int:
        """
        Store an extracted ResumeProfile in PostgreSQL.

        Uses UPSERT (ON CONFLICT) so re-ingesting the same file for the same
        client updates the existing row rather than failing.

        Args:
            profile: Extracted ResumeProfile from Stage 2
            source_file: Original filename
            source_path: Full path to original file
            file_hash: MD5 hash of file content
            client_id: Client identifier (NDA isolation)
            job_id: Job opening identifier

        Returns:
            Row ID of the inserted/updated record

        Raises:
            ValueError: If client_id or job_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for storing profiles (NDA enforcement)")
        if not job_id or not job_id.strip():
            raise ValueError("job_id is required for storing profiles")

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO resume_profiles (
                    client_id, job_id, source_file, source_path, file_hash,
                    first_name, middle_name, last_name, email, phone, location,
                    career_summary, skills, total_experience_years,
                    work_experiences, projects, education, certifications,
                    domain_expertise, raw_text, extracted_at
                ) VALUES (
                    %(client_id)s, %(job_id)s, %(source_file)s, %(source_path)s, %(file_hash)s,
                    %(first_name)s, %(middle_name)s, %(last_name)s, %(email)s, %(phone)s, %(location)s,
                    %(career_summary)s, %(skills)s, %(total_experience_years)s,
                    %(work_experiences)s, %(projects)s, %(education)s, %(certifications)s,
                    %(domain_expertise)s, %(raw_text)s, %(extracted_at)s
                )
                ON CONFLICT (client_id, file_hash) DO UPDATE SET
                    job_id = EXCLUDED.job_id,
                    source_file = EXCLUDED.source_file,
                    source_path = EXCLUDED.source_path,
                    first_name = EXCLUDED.first_name,
                    middle_name = EXCLUDED.middle_name,
                    last_name = EXCLUDED.last_name,
                    email = EXCLUDED.email,
                    phone = EXCLUDED.phone,
                    location = EXCLUDED.location,
                    career_summary = EXCLUDED.career_summary,
                    skills = EXCLUDED.skills,
                    total_experience_years = EXCLUDED.total_experience_years,
                    work_experiences = EXCLUDED.work_experiences,
                    projects = EXCLUDED.projects,
                    education = EXCLUDED.education,
                    certifications = EXCLUDED.certifications,
                    domain_expertise = EXCLUDED.domain_expertise,
                    raw_text = EXCLUDED.raw_text,
                    extracted_at = EXCLUDED.extracted_at
                RETURNING id
            """, {
                "client_id": client_id.strip(),
                "job_id": job_id.strip(),
                "source_file": source_file,
                "source_path": source_path,
                "file_hash": file_hash,
                "first_name": profile.first_name,
                "middle_name": profile.middle_name,
                "last_name": profile.last_name,
                "email": profile.email,
                "phone": profile.phone,
                "location": profile.location,
                "career_summary": profile.career_summary,
                "skills": json.dumps(profile.skills),
                "total_experience_years": profile.total_experience_years,
                "work_experiences": json.dumps([exp.model_dump() for exp in profile.work_experiences]),
                "projects": json.dumps([proj.model_dump() for proj in profile.projects]),
                "education": json.dumps(profile.education),
                "certifications": json.dumps(profile.certifications),
                "domain_expertise": json.dumps(profile.domain_expertise),
                "raw_text": profile.raw_text,
                "extracted_at": datetime.now().isoformat(),
            })
            row = cur.fetchone()
            row_id = row["id"] if row else 0

        self.conn.commit()
        logger.info(f"Stored profile: {profile.full_name} ({source_file}) → client={client_id}, id={row_id}")
        return row_id

    # ─────────────────────────────────────────────────────────────────────────
    # READ OPERATIONS — ALL FILTERED BY CLIENT_ID (NDA ENFORCEMENT)
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_profiles(self, client_id: str) -> list[ResumeProfile]:
        """Retrieve all profiles for a specific client."""
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required (NDA enforcement)")

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM resume_profiles WHERE client_id = %s ORDER BY id",
                (client_id.strip(),)
            )
            rows = cur.fetchall()

        return [self._row_to_profile(row) for row in rows]

    def get_all_profiles_with_metadata(self, client_id: str) -> list[dict]:
        """Retrieve all profiles with source metadata for a client."""
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required (NDA enforcement)")

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM resume_profiles WHERE client_id = %s ORDER BY id",
                (client_id.strip(),)
            )
            rows = cur.fetchall()

        return [
            {
                "profile": self._row_to_profile(row),
                "source_file": row["source_file"],
                "source_path": row["source_path"],
                "file_hash": row["file_hash"],
                "client_id": row["client_id"],
                "job_id": row["job_id"],
                "extracted_at": str(row["extracted_at"]),
            }
            for row in rows
        ]

    def get_profile_count(self, client_id: Optional[str] = None) -> int:
        """Return profile count. If client_id given, scoped to that client."""
        with self.conn.cursor() as cur:
            if client_id:
                cur.execute(
                    "SELECT COUNT(*) as cnt FROM resume_profiles WHERE client_id = %s",
                    (client_id.strip(),)
                )
            else:
                cur.execute("SELECT COUNT(*) as cnt FROM resume_profiles")
            return cur.fetchone()["cnt"]

    def get_ingested_hashes(self, client_id: str) -> set[str]:
        """Return all file hashes for a client."""
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required (NDA enforcement)")

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT file_hash FROM resume_profiles WHERE client_id = %s",
                (client_id.strip(),)
            )
            return {row["file_hash"] for row in cur.fetchall()}

    # ─────────────────────────────────────────────────────────────────────────
    # FILE CHANGE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def needs_processing(self, file_path: str | Path, client_id: str) -> bool:
        """Check if a file needs processing for a client."""
        file_hash = self.compute_file_hash(file_path)
        ingested = self.get_ingested_hashes(client_id)
        return file_hash not in ingested

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        """Compute MD5 hash of a file's content."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    # ─────────────────────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get database status including per-client breakdown."""
        total = self.get_profile_count()

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT client_id, job_id, COUNT(*) as cnt "
                "FROM resume_profiles GROUP BY client_id, job_id "
                "ORDER BY client_id, job_id"
            )
            breakdown = [
                {"client_id": row["client_id"], "job_id": row["job_id"], "count": row["cnt"]}
                for row in cur.fetchall()
            ]

        return {
            "db_type": "PostgreSQL + pgvector",
            "profile_count": total,
            "client_job_breakdown": breakdown,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────────────────────────────

    def _row_to_profile(self, row: dict) -> ResumeProfile:
        """Convert a database row dict to a ResumeProfile object."""
        skills = row["skills"] if isinstance(row["skills"], list) else json.loads(row["skills"] or "[]")
        work_exps_raw = row["work_experiences"] if isinstance(row["work_experiences"], list) else json.loads(row["work_experiences"] or "[]")
        projects_raw = row["projects"] if isinstance(row["projects"], list) else json.loads(row["projects"] or "[]")
        education = row["education"] if isinstance(row["education"], list) else json.loads(row["education"] or "[]")
        certifications = row["certifications"] if isinstance(row["certifications"], list) else json.loads(row["certifications"] or "[]")
        domain_expertise = row["domain_expertise"] if isinstance(row["domain_expertise"], list) else json.loads(row["domain_expertise"] or "[]")

        work_experiences = [WorkExperience(**exp) for exp in work_exps_raw if isinstance(exp, dict)]
        projects = [Project(**proj) for proj in projects_raw if isinstance(proj, dict)]

        return ResumeProfile(
            client_id=row["client_id"],
            job_id=row["job_id"],
            first_name=row["first_name"] or "",
            middle_name=row["middle_name"],
            last_name=row["last_name"] or "",
            email=row["email"],
            phone=row["phone"],
            location=row["location"],
            career_summary=row["career_summary"] or "",
            skills=skills,
            total_experience_years=row["total_experience_years"],
            work_experiences=work_experiences,
            projects=projects,
            education=education,
            certifications=certifications,
            domain_expertise=domain_expertise,
            raw_text=row["raw_text"] or "",
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.debug("PostgreSQL connection closed")
