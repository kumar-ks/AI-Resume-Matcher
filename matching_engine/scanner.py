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
from matching_engine.hallucination_check import check_hallucination
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

            # Hallucination check: verify extracted fields against source text
            hallucination_report = check_hallucination(profile, raw_text)
            if not hallucination_report.is_reliable:
                logger.warning(
                    f"  ⚠️  Hallucination detected for {file_path.name} "
                    f"(confidence={hallucination_report.overall_confidence:.0%})"
                )
                for w in hallucination_report.warnings:
                    logger.warning(f"      {w}")
                print(f"    ⚠️  Grounding confidence: {hallucination_report.overall_confidence:.0%} — some extracted data may be inaccurate")
            else:
                logger.info(f"    ✓ Grounding check passed ({hallucination_report.overall_confidence:.0%})")

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
    """Legacy single-field builder. Kept for backward compat."""
    parts = []
    if profile.career_summary:
        parts.append(profile.career_summary)
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    for exp in profile.work_experiences[:5]:
        if exp.title and exp.company:
            parts.append(f"{exp.title} at {exp.company}")
    if profile.domain_expertise:
        parts.append("Domains: " + ", ".join(profile.domain_expertise))
    return ". ".join(parts)[:2000]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF KEYWORD EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

# Generic filler words that add no signal for matching
_STOPWORDS = {
    "responsible", "responsibilities", "team", "player", "excellent",
    "communication", "skills", "ability", "strong", "good", "great",
    "experience", "working", "work", "worked", "various", "multiple",
    "including", "using", "used", "also", "well", "ensure", "ensured",
    "manage", "managed", "support", "supported", "develop", "developed",
    "involved", "involvement", "knowledge", "understanding", "help",
    "helped", "assist", "assisted", "provide", "provided", "maintain",
    "maintained", "perform", "performed", "create", "created", "implement",
    "implemented", "based", "related", "required", "different", "several",
    "within", "across", "along", "part", "role", "position", "company",
}


def _extract_keywords_tfidf(text: str, top_n: int = 80) -> str:
    """
    Extract top keywords from text using TF-IDF scoring.

    Uses term frequency with inverse document frequency approximation
    to keep domain-specific terms and drop generic ones.

    Args:
        text: Raw text to extract keywords from
        top_n: Number of top keywords to keep

    Returns:
        Space-separated string of top keywords
    """
    import re
    from collections import Counter
    import math

    if not text or not text.strip():
        return ""

    # Tokenize: split on non-alphanumeric, keep words 3+ chars
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b', text.lower())

    # Remove stopwords
    words = [w for w in words if w not in _STOPWORDS]

    if not words:
        return ""

    # Term frequency
    tf = Counter(words)
    total_terms = len(words)

    # IDF approximation: rarer words in this document get higher weight
    # (since we don't have a corpus, we use log(total/freq) as a proxy)
    scored = {}
    for word, count in tf.items():
        tf_score = count / total_terms
        # Penalize very common words (appear > 5% of text)
        idf_approx = math.log(total_terms / (1 + count))
        scored[word] = tf_score * idf_approx

    # Sort by score descending, take top N
    top_keywords = sorted(scored.keys(), key=lambda w: scored[w], reverse=True)[:top_n]

    return " ".join(top_keywords)


def _infer_role_level(profile) -> str:
    """
    Infer seniority/role level from profile data.

    Uses heuristics based on experience years and job titles.

    Returns:
        One of: "entry", "mid", "senior", "lead", "principal"
    """
    years = profile.total_experience_years or 0

    # Check titles for explicit level indicators
    titles = " ".join(
        (exp.title or "").lower() for exp in profile.work_experiences[:3]
    )

    if any(k in titles for k in ["principal", "distinguished", "fellow", "chief", "cto", "vp"]):
        return "principal"
    if any(k in titles for k in ["lead", "head", "director", "architect", "staff"]):
        return "lead"
    if any(k in titles for k in ["senior", "sr.", "sr "]):
        return "senior"
    if any(k in titles for k in ["junior", "jr.", "jr ", "intern", "trainee", "graduate"]):
        return "entry"

    # Fallback to years of experience
    if years >= 12:
        return "lead"
    elif years >= 7:
        return "senior"
    elif years >= 3:
        return "mid"
    else:
        return "entry"


def _build_multi_field_texts(profile) -> tuple[str, str, str]:
    """
    Build 3 curated texts for multi-field embeddings using TF-IDF extraction.

    Returns:
        (skills_text, experience_text, summary_text)
    """
    # ── Skills text: technologies, tools, certifications ──────────────────
    skills_raw_parts = []
    if profile.skills:
        skills_raw_parts.append(", ".join(profile.skills))
    for exp in profile.work_experiences[:5]:
        if exp.technologies:
            skills_raw_parts.append(", ".join(exp.technologies))
    if profile.certifications:
        skills_raw_parts.append(", ".join(profile.certifications))
    skills_raw = ". ".join(skills_raw_parts)
    skills_text = _extract_keywords_tfidf(skills_raw, top_n=60)[:1500]

    # ── Experience text: role titles, companies, domains ───────────────────
    exp_raw_parts = []
    for exp in profile.work_experiences[:7]:
        if exp.title:
            exp_raw_parts.append(exp.title)
        if exp.company:
            exp_raw_parts.append(exp.company)
        if exp.domain:
            exp_raw_parts.append(exp.domain)
        if exp.responsibilities:
            exp_raw_parts.extend(exp.responsibilities[:3])
    if profile.domain_expertise:
        exp_raw_parts.extend(profile.domain_expertise)
    exp_raw = ". ".join(exp_raw_parts)
    experience_text = _extract_keywords_tfidf(exp_raw, top_n=60)[:1500]

    # ── Summary text: career summary, achievements ────────────────────────
    sum_raw_parts = []
    if profile.career_summary:
        sum_raw_parts.append(profile.career_summary)
    if profile.domain_expertise:
        sum_raw_parts.append(", ".join(profile.domain_expertise))
    if profile.education:
        sum_raw_parts.append(", ".join(profile.education[:3]))
    sum_raw = ". ".join(sum_raw_parts)
    summary_text = _extract_keywords_tfidf(sum_raw, top_n=50)[:1500]

    return skills_text, experience_text, summary_text
