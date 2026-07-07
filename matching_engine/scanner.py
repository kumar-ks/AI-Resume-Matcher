"""
Async Resume Scanner
=====================

Scans resume files, extracts profiles via LLM, and stores them in
the database + vector store. Runs asynchronously (non-blocking).

MULTI-TENANT:
    - Requires client_id and job_id for every ingest operation
    - All stored profiles and embeddings are tagged with these identifiers
    - Deduplication is scoped to the client (same file can exist under different clients)

RESPONSIBILITIES:
    - Detect new/modified resume files in the resumes folder
    - Extract profiles via the existing Stage 2 (ResumeUnderstanding)
    - Store extracted profiles in SQLite (database.py) with client_id/job_id
    - Store embeddings in ChromaDB (vector_store.py) with client_id/job_id
    - Copy processed resume files to scanned_files/
    - Skip files already processed for this client (by file hash)

CALLED BY:
    - run.py → --ingest mode (explicit scan)
    - run.py → default "db_first" mode (scans for new files after DB query)

ASYNC:
    The scanner is an async function — it can run concurrently with other
    operations without blocking the main thread.
"""

import logging
import time
from pathlib import Path

from matching_engine.database import ProfileDatabase
from matching_engine.file_loader import extract_text, SUPPORTED_EXTENSIONS
from matching_engine.resume_understanding import ResumeUnderstanding
from matching_engine.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def scan_and_ingest(
    resumes_dir: str | Path,
    db: ProfileDatabase,
    vector_store: VectorStore,
    client_id: str,
    job_id: str,
    model: str = "ollama/llama3",
    temperature: float = 0.1,
) -> dict:
    """
    Scan a directory for resume files and ingest new/modified ones into the DB.

    Args:
        resumes_dir: Path to the folder containing resume files
        db: ProfileDatabase instance (SQLite)
        vector_store: VectorStore instance (ChromaDB)
        client_id: Client identifier (NDA isolation boundary) — REQUIRED
        job_id: Job opening identifier — REQUIRED
        model: LLM model to use for extraction (from config)
        temperature: LLM temperature

    Returns:
        dict with new_count, skipped_count, failed_count, total_in_db,
        elapsed_seconds, client_id, job_id

    Raises:
        ValueError: If client_id or job_id is empty
    """
    if not client_id or not client_id.strip():
        raise ValueError("client_id is required for ingest (NDA enforcement)")
    if not job_id or not job_id.strip():
        raise ValueError("job_id is required for ingest")

    start_time = time.time()
    resumes_dir = Path(resumes_dir)

    if not resumes_dir.exists():
        logger.error(f"Resumes directory not found: {resumes_dir}")
        return {
            "new_count": 0, "skipped_count": 0, "failed_count": 0,
            "total_in_db": 0, "elapsed_seconds": 0,
            "client_id": client_id, "job_id": job_id,
        }

    # ── Step 1: Find all supported files ──────────────────────────────────────
    all_files = sorted(
        f for f in resumes_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logger.info(f"Scanner found {len(all_files)} resume files in {resumes_dir}")

    # ── Step 2: Determine which files need processing (scoped to client) ──────
    ingested_hashes = db.get_ingested_hashes(client_id)
    files_to_process = []
    skipped = 0

    for file_path in all_files:
        file_hash = ProfileDatabase.compute_file_hash(file_path)
        if file_hash in ingested_hashes:
            skipped += 1
            logger.debug(f"Skipping (already in DB for client={client_id}): {file_path.name}")
        else:
            files_to_process.append((file_path, file_hash))

    logger.info(
        f"Scanner: {len(files_to_process)} new files to process for "
        f"client={client_id}, job={job_id}. {skipped} already in DB."
    )

    if not files_to_process:
        return {
            "new_count": 0,
            "skipped_count": skipped,
            "failed_count": 0,
            "total_in_db": db.get_profile_count(client_id),
            "elapsed_seconds": round(time.time() - start_time, 1),
            "client_id": client_id,
            "job_id": job_id,
        }

    # ── Step 3: Process each new file ─────────────────────────────────────────
    resume_extractor = ResumeUnderstanding(model=model, temperature=temperature)
    new_count = 0
    failed_count = 0

    for i, (file_path, file_hash) in enumerate(files_to_process, 1):
        logger.info(f"Ingesting [{i}/{len(files_to_process)}]: {file_path.name}")
        print(f"  Ingesting [{i}/{len(files_to_process)}]: {file_path.name}...")

        try:
            # Extract text from file
            raw_text = extract_text(file_path)
            if not raw_text.strip():
                logger.warning(f"Empty text extracted from {file_path.name}, skipping")
                failed_count += 1
                continue

            # Extract structured profile via LLM (Stage 2)
            profile = await resume_extractor.extract(raw_text)

            # Tag profile with client_id and job_id
            profile.client_id = client_id.strip()
            profile.job_id = job_id.strip()

            # Quality check
            has_name = bool(profile.full_name.strip())
            has_skills = len(profile.skills) > 0
            has_experience = profile.total_experience_years is not None

            if not has_name and not has_skills and not has_experience:
                logger.warning(
                    f"  ⚠️  Low-quality extraction for {file_path.name} "
                    f"(no name, no skills, no experience). Storing with raw text only."
                )

            # Store in SQLite (with client_id and job_id)
            db.store_profile(
                profile=profile,
                source_file=file_path.name,
                source_path=str(file_path),
                file_hash=file_hash,
                client_id=client_id.strip(),
                job_id=job_id.strip(),
            )

            # Build embedding text
            embed_text = _build_embedding_text(profile)

            # Store embedding in ChromaDB (with client_id and job_id)
            vector_store.store_embedding(
                file_hash=file_hash,
                text=embed_text,
                client_id=client_id.strip(),
                job_id=job_id.strip(),
                metadata={
                    "source_file": file_path.name,
                    "full_name": profile.full_name,
                    "experience_years": profile.total_experience_years or 0,
                },
            )

            new_count += 1
            logger.info(f"  ✓ Ingested: {profile.full_name} ({file_path.name})")

        except Exception as e:
            failed_count += 1
            logger.error(f"  ✗ Failed to ingest {file_path.name}: {e}")
            print(f"  ✗ Failed: {file_path.name} ({type(e).__name__})")

    # ── Step 4: Report results ────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    total_in_db = db.get_profile_count(client_id)

    logger.info(
        f"Scanner complete: {new_count} new, {skipped} skipped, "
        f"{failed_count} failed. Total in DB for client={client_id}: {total_in_db}. Time: {elapsed}s"
    )

    return {
        "new_count": new_count,
        "skipped_count": skipped,
        "failed_count": failed_count,
        "total_in_db": total_in_db,
        "elapsed_seconds": elapsed,
        "client_id": client_id,
        "job_id": job_id,
    }


def _build_embedding_text(profile) -> str:
    """
    Build a representative text for embedding storage.

    Combines the most semantically meaningful fields:
    - Career summary (high-level description)
    - Skills (searchable keywords)
    - Job titles + companies (role context)
    - Domain expertise
    """
    parts = []

    if profile.career_summary:
        parts.append(profile.career_summary)

    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))

    for exp in profile.work_experiences[:5]:
        if exp.title and exp.company:
            parts.append(f"{exp.title} at {exp.company}")
        if exp.technologies:
            parts.append("Technologies: " + ", ".join(exp.technologies[:10]))

    if profile.domain_expertise:
        parts.append("Domains: " + ", ".join(profile.domain_expertise))

    if profile.certifications:
        parts.append("Certifications: " + ", ".join(profile.certifications))

    text = ". ".join(parts)
    return text[:2000]
