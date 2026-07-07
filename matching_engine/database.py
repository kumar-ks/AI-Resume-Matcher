"""
Database Layer — SQLite Profile Storage
=========================================

Manages persistent storage of extracted resume profiles in SQLite.
Each resume is processed once (via LLM) and stored; subsequent matches
read from DB instead of re-extracting.

MULTI-TENANT ISOLATION (NDA ENFORCEMENT):
    - Every profile is tagged with client_id and job_id
    - ALL read queries MUST filter by client_id
    - Resumes under one client_id can NEVER be returned for another client
    - Within the same client_id, resumes are shared across job_ids freely

SCHEMA:
    resume_profiles:
        id              INTEGER PRIMARY KEY
        client_id       TEXT NOT NULL  (NDA isolation boundary)
        job_id          TEXT NOT NULL  (job opening identifier)
        source_file     TEXT    (original filename)
        source_path     TEXT    (original full path)
        file_hash       TEXT    (MD5 of file content — detect changes)
        scanned_copy    TEXT    (path to copy in data/scanned_files/)
        first_name      TEXT
        middle_name     TEXT
        last_name       TEXT
        email           TEXT
        phone           TEXT
        location        TEXT
        career_summary  TEXT
        skills          TEXT    (JSON array)
        total_experience_years REAL
        work_experiences TEXT   (JSON array of objects)
        projects        TEXT    (JSON array)
        education       TEXT    (JSON array)
        certifications  TEXT    (JSON array)
        domain_expertise TEXT   (JSON array)
        raw_text        TEXT
        extracted_at    TEXT    (ISO timestamp)

    UNIQUE CONSTRAINT: (client_id, file_hash)
        Same file CAN exist under different clients (separate ingest).
        Same file CANNOT be duplicated within the same client.

CALLED BY:
    - run.py → --ingest mode (writes profiles)
    - run.py → --match mode (reads profiles)
    - matching_engine.scanner (async background scanner)
"""

import hashlib
import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from matching_engine.models import (
    Project,
    ResumeProfile,
    WorkExperience,
)

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("data/profiles.db")
DEFAULT_SCANNED_FILES_PATH = Path("data/scanned_files")


class ProfileDatabase:
    """
    SQLite-based storage for extracted resume profiles with client isolation.

    STRICT RULE: All read operations require a client_id parameter.
    Profiles from one client are NEVER visible to another client.

    Usage:
        db = ProfileDatabase(db_path="data/profiles.db")
        db.store_profile(profile, source_file, source_path, file_hash, client_id="C1", job_id="J1")
        profiles = db.get_all_profiles(client_id="C1")
        is_new = db.needs_processing(file_path, client_id="C1")
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        scanned_files_path: str | Path = DEFAULT_SCANNED_FILES_PATH,
    ):
        """
        Initialize database connection and ensure tables exist.

        Args:
            db_path: Path to SQLite database file (created if not exists)
            scanned_files_path: Directory to store copies of processed resumes
        """
        self.db_path = Path(db_path)
        self.scanned_files_path = Path(scanned_files_path)

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.scanned_files_path.mkdir(parents=True, exist_ok=True)

        # Connect and create tables
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

        logger.info(f"ProfileDatabase initialized: {self.db_path}")

    def _create_tables(self) -> None:
        """Create the resume_profiles table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS resume_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                source_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                scanned_copy TEXT,
                first_name TEXT DEFAULT '',
                middle_name TEXT,
                last_name TEXT DEFAULT '',
                email TEXT,
                phone TEXT,
                location TEXT,
                career_summary TEXT DEFAULT '',
                skills TEXT DEFAULT '[]',
                total_experience_years REAL,
                work_experiences TEXT DEFAULT '[]',
                projects TEXT DEFAULT '[]',
                education TEXT DEFAULT '[]',
                certifications TEXT DEFAULT '[]',
                domain_expertise TEXT DEFAULT '[]',
                raw_text TEXT DEFAULT '',
                extracted_at TEXT NOT NULL,
                UNIQUE(client_id, file_hash)
            )
        """)
        # Indexes for fast client-scoped queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiles_client_id
            ON resume_profiles(client_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiles_client_job
            ON resume_profiles(client_id, job_id)
        """)
        self.conn.commit()
        logger.debug("Database tables verified/created")

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
        Store an extracted ResumeProfile in the database.

        Also copies the original resume file to scanned_files/ for reference.

        Args:
            profile: Extracted ResumeProfile from Stage 2
            source_file: Original filename (e.g., "resume.pdf")
            source_path: Full path to original file
            file_hash: MD5 hash of file content
            client_id: Client identifier (NDA isolation boundary) — REQUIRED
            job_id: Job opening identifier — REQUIRED

        Returns:
            Row ID of the inserted record

        Raises:
            ValueError: If client_id or job_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for storing profiles (NDA enforcement)")
        if not job_id or not job_id.strip():
            raise ValueError("job_id is required for storing profiles")

        # Copy the resume file to scanned_files/
        scanned_copy = self._copy_to_scanned(source_path, file_hash, source_file)

        # Serialize complex fields to JSON
        data = (
            client_id.strip(),
            job_id.strip(),
            source_file,
            source_path,
            file_hash,
            str(scanned_copy) if scanned_copy else None,
            profile.first_name,
            profile.middle_name,
            profile.last_name,
            profile.email,
            profile.phone,
            profile.location,
            profile.career_summary,
            json.dumps(profile.skills),
            profile.total_experience_years,
            json.dumps([exp.model_dump() for exp in profile.work_experiences]),
            json.dumps([proj.model_dump() for proj in profile.projects]),
            json.dumps(profile.education),
            json.dumps(profile.certifications),
            json.dumps(profile.domain_expertise),
            profile.raw_text,
            datetime.now().isoformat(),
        )

        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO resume_profiles (
                client_id, job_id,
                source_file, source_path, file_hash, scanned_copy,
                first_name, middle_name, last_name, email, phone, location,
                career_summary, skills, total_experience_years,
                work_experiences, projects, education, certifications,
                domain_expertise, raw_text, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()

        logger.info(
            f"Stored profile: {profile.full_name} ({source_file}) "
            f"→ client={client_id}, job={job_id}, id={cursor.lastrowid}"
        )
        return cursor.lastrowid

    def _copy_to_scanned(self, source_path: str, file_hash: str, filename: str) -> Optional[Path]:
        """Copy the original resume file to scanned_files/ directory."""
        try:
            src = Path(source_path)
            if not src.exists():
                return None
            dest = self.scanned_files_path / f"{file_hash[:8]}_{filename}"
            if not dest.exists():
                shutil.copy2(str(src), str(dest))
                logger.debug(f"Copied resume to: {dest}")
            return dest
        except Exception as e:
            logger.warning(f"Failed to copy resume to scanned_files: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # READ OPERATIONS — ALL FILTERED BY CLIENT_ID (NDA ENFORCEMENT)
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_profiles(self, client_id: str) -> list[ResumeProfile]:
        """
        Retrieve all stored profiles for a specific client.

        NDA ENFORCEMENT: Only returns profiles belonging to the given client_id.

        Args:
            client_id: The client whose profiles to retrieve

        Returns:
            List of ResumeProfile objects (deserialized from DB)

        Raises:
            ValueError: If client_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for retrieving profiles (NDA enforcement)")

        cursor = self.conn.execute(
            "SELECT * FROM resume_profiles WHERE client_id = ? ORDER BY id",
            (client_id.strip(),)
        )
        rows = cursor.fetchall()
        profiles = [self._row_to_profile(row) for row in rows]
        logger.debug(f"Retrieved {len(profiles)} profiles for client={client_id}")
        return profiles

    def get_all_profiles_with_metadata(self, client_id: str) -> list[dict]:
        """
        Retrieve all profiles with source file metadata for a specific client.

        NDA ENFORCEMENT: Only returns profiles belonging to the given client_id.

        Args:
            client_id: The client whose profiles to retrieve

        Returns:
            List of dicts with 'profile', 'source_file', 'source_path', 'file_hash', etc.

        Raises:
            ValueError: If client_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for retrieving profiles (NDA enforcement)")

        cursor = self.conn.execute(
            "SELECT * FROM resume_profiles WHERE client_id = ? ORDER BY id",
            (client_id.strip(),)
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "profile": self._row_to_profile(row),
                "source_file": row["source_file"],
                "source_path": row["source_path"],
                "file_hash": row["file_hash"],
                "client_id": row["client_id"],
                "job_id": row["job_id"],
                "extracted_at": row["extracted_at"],
            })
        return results

    def get_profile_count(self, client_id: Optional[str] = None) -> int:
        """
        Return number of profiles in the database.

        Args:
            client_id: If provided, count only for this client.
                       If None, returns total count across ALL clients (for status display).
        """
        if client_id:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM resume_profiles WHERE client_id = ?",
                (client_id.strip(),)
            )
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM resume_profiles")
        return cursor.fetchone()[0]

    def get_ingested_hashes(self, client_id: str) -> set[str]:
        """
        Return set of all file hashes already in the database for a client.

        NDA ENFORCEMENT: Only returns hashes belonging to the given client_id.

        Args:
            client_id: The client to check ingested files for

        Raises:
            ValueError: If client_id is empty
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id is required for checking ingested hashes (NDA enforcement)")

        cursor = self.conn.execute(
            "SELECT file_hash FROM resume_profiles WHERE client_id = ?",
            (client_id.strip(),)
        )
        return {row[0] for row in cursor.fetchall()}

    # ─────────────────────────────────────────────────────────────────────────
    # FILE CHANGE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def needs_processing(self, file_path: str | Path, client_id: str) -> bool:
        """
        Check if a file needs to be (re-)processed for a specific client.

        Returns True if:
            - File hash not in database for this client (new file)
            - File content has changed (hash mismatch)

        Note: The same file CAN exist under different clients (separate ingest).

        Args:
            file_path: Path to the resume file to check
            client_id: The client context for this check

        Returns:
            True if the file should be processed, False if already in DB for this client
        """
        file_hash = self.compute_file_hash(file_path)
        ingested = self.get_ingested_hashes(client_id)
        needs = file_hash not in ingested
        logger.debug(
            f"needs_processing({Path(file_path).name}, client={client_id}): "
            f"hash={file_hash[:8]}... → {needs}"
        )
        return needs

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
        """Get database status information including per-client breakdown."""
        total_count = self.get_profile_count()
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        scanned_count = len(list(self.scanned_files_path.glob("*"))) if self.scanned_files_path.exists() else 0

        # Per-client breakdown
        cursor = self.conn.execute(
            "SELECT client_id, job_id, COUNT(*) as cnt "
            "FROM resume_profiles GROUP BY client_id, job_id ORDER BY client_id, job_id"
        )
        client_job_breakdown = [
            {"client_id": row[0], "job_id": row[1], "count": row[2]}
            for row in cursor.fetchall()
        ]

        return {
            "db_path": str(self.db_path),
            "profile_count": total_count,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "scanned_files_count": scanned_count,
            "scanned_files_path": str(self.scanned_files_path),
            "client_job_breakdown": client_job_breakdown,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: Row → ResumeProfile conversion
    # ─────────────────────────────────────────────────────────────────────────

    def _row_to_profile(self, row: sqlite3.Row) -> ResumeProfile:
        """Convert a database row to a ResumeProfile object."""
        # Deserialize work experiences
        work_exps_raw = json.loads(row["work_experiences"] or "[]")
        work_experiences = [
            WorkExperience(**exp) for exp in work_exps_raw if isinstance(exp, dict)
        ]

        # Deserialize projects
        projects_raw = json.loads(row["projects"] or "[]")
        projects = [
            Project(**proj) for proj in projects_raw if isinstance(proj, dict)
        ]

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
            skills=json.loads(row["skills"] or "[]"),
            total_experience_years=row["total_experience_years"],
            work_experiences=work_experiences,
            projects=projects,
            education=json.loads(row["education"] or "[]"),
            certifications=json.loads(row["certifications"] or "[]"),
            domain_expertise=json.loads(row["domain_expertise"] or "[]"),
            raw_text=row["raw_text"] or "",
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.debug("Database connection closed")
