"""
FastAPI Server — AI Resume Matcher API
========================================

ENDPOINTS:
    POST /api/ingest    — Upload resumes + JD, ingest resumes, match against JD, save results
    POST /api/match     — Upload JD, match against existing profiles, save results
    GET  /api/status    — Get unsent results for a client_id + job_id
    GET  /health        — Health check (no auth)

AUTH:
    All endpoints (except /health) require X-API-Key header.

USAGE:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from api.auth import verify_api_key


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — writes to logs/api.log with daily rollover
# ─────────────────────────────────────────────────────────────────────────────

def _setup_api_logging():
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "api.log"

    log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        filename=str(log_file), when="midnight", interval=1,
        backupCount=30, encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    for name in ["httpx", "sentence_transformers", "huggingface_hub", "LiteLLM", "litellm"]:
        logging.getLogger(name).setLevel(logging.WARNING)


_setup_api_logging()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Resume Matcher API",
    description="Multi-tenant resume matching engine with NDA-level client isolation",
    version="3.0.0",
)

UPLOAD_BASE = Path("data/uploads")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE SETUP (match_results in PostgreSQL)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_results_table():
    """Create match_results table if it doesn't exist."""
    import psycopg
    from psycopg.rows import dict_row

    db_url = os.environ.get("DATABASE_URL", "postgresql://matcher:matcher_secret@localhost:5432/resume_matcher")
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS "pgcrypto";
                CREATE TABLE IF NOT EXISTS match_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    client_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    resume_file_hash TEXT NOT NULL,
                    full_name TEXT,
                    email TEXT,
                    phone TEXT,
                    total_experience_years REAL,
                    qualification_percentage REAL,
                    recommendation TEXT,
                    reasoning TEXT,
                    key_strengths JSONB DEFAULT '[]',
                    missing_skills JSONB DEFAULT '[]',
                    top_skills JSONB DEFAULT '[]',
                    scoring_breakdown JSONB DEFAULT '{}',
                    matched_at TIMESTAMPTZ DEFAULT NOW(),
                    is_delivered BOOLEAN DEFAULT FALSE,
                    UNIQUE(client_id, job_id, resume_file_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_results_client_job
                ON match_results(client_id, job_id);
                CREATE INDEX IF NOT EXISTS idx_results_undelivered
                ON match_results(client_id, job_id) WHERE is_delivered = FALSE;
            """)
        conn.commit()
    logger.info("match_results table verified/created")


@app.on_event("startup")
async def startup():
    _ensure_results_table()
    _ensure_ollama_running()


def _ensure_ollama_running():
    """Check if Ollama is running. If not, start it automatically."""
    import shutil
    import subprocess
    import time

    model = os.environ.get("MODEL", "ollama/llama3")
    if not model.startswith("ollama/"):
        return

    ollama_path = shutil.which("ollama")
    if not ollama_path:
        logger.warning("Ollama binary not found. LLM calls will fail.")
        return

    # Check if already running
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("Ollama is already running")
            return
    except Exception:
        pass

    # Start it
    logger.info("Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait up to 10 seconds
    for i in range(10):
        time.sleep(1)
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                logger.info(f"Ollama started successfully (took {i+1}s)")
                return
        except Exception:
            continue

    logger.warning("Could not start Ollama after 10s. LLM calls may fail.")


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/ingest — Upload resumes + JD, ingest, match, save results
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_and_match(
    background_tasks: BackgroundTasks,
    client_id: str = Form(..., description="Client identifier (NDA isolation)"),
    job_id: str = Form(..., description="Job opening identifier"),
    jd_file: UploadFile = File(..., description="Job Description file (PDF/DOCX/TXT)"),
    files: list[UploadFile] = File(..., description="Resume files (PDF/DOCX/TXT)"),
):
    """
    Upload resumes + JD. Ingests resumes, matches against JD, saves results.
    Returns immediately (202). Processing runs in background.
    """
    if not client_id or not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")
    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="job_id is required")
    if not files:
        raise HTTPException(status_code=400, detail="At least one resume file is required")

    # Save JD file
    jd_dir = UPLOAD_BASE / client_id.strip() / job_id.strip() / "jd"
    jd_dir.mkdir(parents=True, exist_ok=True)
    jd_content = await jd_file.read()
    jd_path = jd_dir / (jd_file.filename or "jd.txt")
    with open(jd_path, "wb") as f:
        f.write(jd_content)

    # Save resume files
    resume_dir = UPLOAD_BASE / client_id.strip() / job_id.strip() / "resumes"
    resume_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        content = await file.read()
        dest = resume_dir / (file.filename or f"resume_{uuid.uuid4().hex[:8]}")
        with open(dest, "wb") as f:
            f.write(content)

    logger.info(f"/api/ingest: {len(files)} resumes + JD for client={client_id}, job={job_id}")

    # Run ingest + match in background
    background_tasks.add_task(
        _background_ingest_and_match,
        client_id.strip(), job_id.strip(), str(resume_dir), str(jd_path)
    )

    return JSONResponse(status_code=202, content={
        "message": f"Ingest started. {len(files)} resumes + JD queued for processing.",
        "client_id": client_id.strip(),
        "job_id": job_id.strip(),
        "files_received": len(files),
    })


async def _background_ingest_and_match(client_id: str, job_id: str, resume_dir: str, jd_path: str):
    """Background task: ingest resumes, then match against JD, save results."""
    from matching_engine.database import ProfileDatabase
    from matching_engine.vector_store import VectorStore
    from matching_engine.scanner import scan_and_ingest
    from matching_engine.file_loader import extract_text

    try:
        # Step 1: Ingest resumes
        db = ProfileDatabase()
        vs = VectorStore()

        result = await scan_and_ingest(
            resumes_dir=resume_dir,
            db=db,
            vector_store=vs,
            client_id=client_id,
            job_id=job_id,
            model=os.environ.get("MODEL", "ollama/llama3"),
            temperature=0.1,
        )
        logger.info(f"Ingest done: {result['new_count']} new, {result['skipped_count']} skipped")
        db.close()
        vs.close()

        # Step 2: Match against JD
        jd_text = extract_text(Path(jd_path))
        if jd_text and jd_text.strip():
            await _run_match_and_save(client_id, job_id, jd_text)

    except Exception as e:
        logger.error(f"Background ingest+match failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/match — Upload JD, match against existing profiles, save results
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/match", dependencies=[Depends(verify_api_key)])
async def match_jd(
    background_tasks: BackgroundTasks,
    client_id: str = Form(..., description="Client identifier (NDA isolation)"),
    job_id: str = Form(..., description="Job opening identifier"),
    jd_file: UploadFile = File(..., description="Job Description file (PDF/DOCX/TXT)"),
):
    """
    Match JD against already-ingested profiles. Saves results to DB.
    Returns immediately (202). Processing runs in background.
    """
    if not client_id or not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")
    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="job_id is required")

    # Save JD temporarily
    jd_dir = UPLOAD_BASE / client_id.strip() / job_id.strip() / "jd"
    jd_dir.mkdir(parents=True, exist_ok=True)
    jd_content = await jd_file.read()
    jd_path = jd_dir / (jd_file.filename or "jd.txt")
    with open(jd_path, "wb") as f:
        f.write(jd_content)

    logger.info(f"/api/match: JD uploaded for client={client_id}, job={job_id}")

    # Run match in background
    background_tasks.add_task(
        _background_match, client_id.strip(), job_id.strip(), str(jd_path)
    )

    return JSONResponse(status_code=202, content={
        "message": "Match started. Results will be available via /api/status.",
        "client_id": client_id.strip(),
        "job_id": job_id.strip(),
    })


async def _background_match(client_id: str, job_id: str, jd_path: str):
    """Background task: match JD against existing profiles, save results."""
    from matching_engine.file_loader import extract_text

    try:
        jd_text = extract_text(Path(jd_path))
        if not jd_text or not jd_text.strip():
            logger.error(f"Could not extract text from JD: {jd_path}")
            return

        await _run_match_and_save(client_id, job_id, jd_text)

    except Exception as e:
        logger.error(f"Background match failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: Run matching pipeline and save results to match_results table
# ─────────────────────────────────────────────────────────────────────────────

async def _run_match_and_save(client_id: str, job_id: str, jd_text: str):
    """Run the matching pipeline and persist results to match_results table."""
    import psycopg
    from psycopg.rows import dict_row
    from matching_engine.database import ProfileDatabase
    from matching_engine.vector_store import VectorStore
    from matching_engine.jd_understanding import JDUnderstanding
    from matching_engine.scoring import Scorer
    from matching_engine.semantic_matching import SemanticMatcher
    from matching_engine.explainability import ExplainabilityEngine
    from matching_engine.models import MatchResult, ExplainabilityReport

    db = ProfileDatabase()
    vs = VectorStore()

    profile_count = db.get_profile_count(client_id)
    if profile_count == 0:
        logger.warning(f"No profiles for client={client_id}, skipping match")
        db.close()
        vs.close()
        return

    # Stage 1: Extract JD requirements
    model = os.environ.get("MODEL", "ollama/llama3")
    jd_understanding = JDUnderstanding(model=model, temperature=0.1)
    jd = await jd_understanding.extract(jd_text)
    jd.client_id = client_id
    jd.job_id = job_id

    # Vector search (client-scoped)
    top_n_query = min(50, profile_count)
    similar_results = vs.query_similar(jd_text, client_id=client_id, top_n=top_n_query)

    if similar_results:
        top_hashes = {r["file_hash"] for r in similar_results}
        all_profiles = db.get_all_profiles_with_metadata(client_id)
        candidates = [p for p in all_profiles if p["file_hash"] in top_hashes]
    else:
        candidates = db.get_all_profiles_with_metadata(client_id)

    # Stages 3-4: Score
    scorer = Scorer()
    semantic_matcher = SemanticMatcher()

    results: list[MatchResult] = []
    for cand in candidates:
        profile = cand["profile"]
        semantic_result = semantic_matcher.match(jd, profile)
        scoring = scorer.score(jd, profile, semantic_result)
        results.append(MatchResult(
            candidate=profile,
            job_description=jd,
            qualification_percentage=scoring.qualification_percentage,
            semantic_scores=semantic_result,
            scoring_breakdown=scoring,
            explainability=ExplainabilityReport(),
            key_strengths=[], missing_skills=[], reasoning="", recommendation="",
        ))

    results.sort(key=lambda r: r.qualification_percentage, reverse=True)

    # Stage 5: Explain top 5
    explainability = ExplainabilityEngine(model=model, temperature=0.3)
    for result in results[:5]:
        explanation = await explainability.explain(
            jd, result.candidate, result.scoring_breakdown, result.semantic_scores
        )
        result.key_strengths = explanation.matched_strengths
        result.missing_skills = explanation.missing_skills
        result.reasoning = explanation.reason_for_score
        result.recommendation = explanation.recommendation

    for result in results[5:]:
        explanation = explainability._fallback_explanation(
            jd, result.candidate, result.scoring_breakdown
        )
        result.key_strengths = explanation.matched_strengths
        result.missing_skills = explanation.missing_skills
        result.reasoning = explanation.reason_for_score
        result.recommendation = explanation.recommendation

    # Save results to match_results table
    db_url = os.environ.get("DATABASE_URL", "postgresql://matcher:matcher_secret@localhost:5432/resume_matcher")
    conn = psycopg.connect(db_url, row_factory=dict_row)

    # Build resume_file_hash lookup
    hash_lookup = {cand["profile"].raw_text: cand["file_hash"] for cand in candidates}

    for r in results:
        resume_file_hash = hash_lookup.get(r.candidate.raw_text, "")
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO match_results (
                    client_id, job_id, resume_file_hash, full_name, email, phone,
                    total_experience_years, qualification_percentage,
                    recommendation, reasoning, key_strengths, missing_skills,
                    top_skills, scoring_breakdown
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (client_id, job_id, resume_file_hash) DO UPDATE SET
                    qualification_percentage = EXCLUDED.qualification_percentage,
                    recommendation = EXCLUDED.recommendation,
                    reasoning = EXCLUDED.reasoning,
                    key_strengths = EXCLUDED.key_strengths,
                    missing_skills = EXCLUDED.missing_skills,
                    top_skills = EXCLUDED.top_skills,
                    scoring_breakdown = EXCLUDED.scoring_breakdown,
                    matched_at = NOW(),
                    is_delivered = FALSE
            """, (
                client_id, job_id, resume_file_hash,
                r.candidate.full_name, r.candidate.email, r.candidate.phone,
                r.candidate.total_experience_years, r.qualification_percentage,
                r.recommendation, r.reasoning,
                json.dumps(r.key_strengths), json.dumps(r.missing_skills),
                json.dumps(r.candidate.skills[:10]),
                json.dumps({
                    "must_have_match": round(r.scoring_breakdown.must_have_match, 3),
                    "experience_match": round(r.scoring_breakdown.experience_match, 3),
                    "skills_depth": round(r.scoring_breakdown.skills_depth, 3),
                    "project_relevance": round(r.scoring_breakdown.project_relevance, 3),
                    "recency_factor": round(r.scoring_breakdown.recency_factor, 3),
                }),
            ))
    conn.commit()
    conn.close()
    db.close()
    vs.close()

    logger.info(f"Match complete: {len(results)} results saved for client={client_id}, job={job_id}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/status — Return unsent results for client_id + job_id
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status", dependencies=[Depends(verify_api_key)])
async def get_results(
    client_id: str,
    job_id: str,
):
    """
    Get match results that haven't been delivered yet.
    Marks them as delivered after returning.
    """
    if not client_id or not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")
    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="job_id is required")

    import psycopg
    from psycopg.rows import dict_row

    db_url = os.environ.get("DATABASE_URL", "postgresql://matcher:matcher_secret@localhost:5432/resume_matcher")
    conn = psycopg.connect(db_url, row_factory=dict_row)

    # Fetch undelivered results
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, resume_file_hash, full_name, email, phone,
                   total_experience_years, qualification_percentage,
                   recommendation, reasoning, key_strengths, missing_skills,
                   top_skills, scoring_breakdown, matched_at
            FROM match_results
            WHERE client_id = %s AND job_id = %s AND is_delivered = FALSE
            ORDER BY qualification_percentage DESC
        """, (client_id.strip(), job_id.strip()))
        rows = cur.fetchall()

    if not rows:
        conn.close()
        return {
            "client_id": client_id.strip(),
            "job_id": job_id.strip(),
            "total_results": 0,
            "results": [],
        }

    # Build response
    results = []
    result_ids = []
    for row in rows:
        result_ids.append(row["id"])
        results.append({
            "result_id": str(row["id"]),
            "resume_file_hash": row["resume_file_hash"],
            "full_name": row["full_name"],
            "email": row["email"],
            "phone": row["phone"],
            "total_experience_years": row["total_experience_years"],
            "qualification_percentage": row["qualification_percentage"],
            "recommendation": row["recommendation"],
            "reasoning": row["reasoning"],
            "key_strengths": row["key_strengths"] if isinstance(row["key_strengths"], list) else json.loads(row["key_strengths"] or "[]"),
            "missing_skills": row["missing_skills"] if isinstance(row["missing_skills"], list) else json.loads(row["missing_skills"] or "[]"),
            "top_skills": row["top_skills"] if isinstance(row["top_skills"], list) else json.loads(row["top_skills"] or "[]"),
            "scoring_breakdown": row["scoring_breakdown"] if isinstance(row["scoring_breakdown"], dict) else json.loads(row["scoring_breakdown"] or "{}"),
            "matched_at": row["matched_at"].isoformat() if row["matched_at"] else None,
        })

    # Mark as delivered
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE match_results SET is_delivered = TRUE WHERE id = ANY(%s)",
            (result_ids,)
        )
    conn.commit()
    conn.close()

    return {
        "client_id": client_id.strip(),
        "job_id": job_id.strip(),
        "total_results": len(results),
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/template — Upload a DOCX template for a client
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/template", dependencies=[Depends(verify_api_key)])
async def upload_template(
    client_id: str = Form(..., description="Client identifier"),
    template_file: UploadFile = File(..., description="DOCX template file"),
):
    """
    Upload a DOCX template for a client.

    The latest uploaded template always wins — previous templates are replaced.
    Templates are stored at: data/templates/{client_id}/template.docx
    """
    if not client_id or not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")

    filename = template_file.filename or "template.docx"
    if not filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Template must be a .docx file")

    # Store template (overwrite existing = latest always wins)
    template_dir = Path("data/templates") / client_id.strip()
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / "template.docx"

    content = await template_file.read()
    with open(template_path, "wb") as f:
        f.write(content)

    logger.info(f"Template uploaded for client={client_id}: {template_path}")

    return {
        "message": "Template uploaded successfully",
        "client_id": client_id.strip(),
        "template_file": filename,
        "stored_at": str(template_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/generate-doc — Convert candidate resume into client template
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/generate-doc", dependencies=[Depends(verify_api_key)])
async def generate_document(
    client_id: str = Form(..., description="Client identifier"),
    resume_file_hash: str = Form(..., description="MD5 hash of the candidate's resume file"),
):
    """
    Convert a candidate's profile into the client's DOCX template.

    Requirements:
        - Client must have uploaded a template via POST /api/template
        - Candidate must exist in the DB (identified by resume_file_hash)

    Returns:
        The generated DOCX file as a download.
    """
    from fastapi.responses import FileResponse
    from matching_engine.database import ProfileDatabase
    from matching_engine.template_renderer import _render_single

    if not client_id or not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")
    if not resume_file_hash or not resume_file_hash.strip():
        raise HTTPException(status_code=400, detail="resume_file_hash is required")

    # Find the client's template
    template_path = Path("data/templates") / client_id.strip() / "template.docx"
    if not template_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No template found for client '{client_id}'. Upload one via POST /api/template."
        )

    # Find the candidate's profile
    db = ProfileDatabase()
    all_profiles = db.get_all_profiles_with_metadata(client_id.strip())
    db.close()

    profile_data = None
    for p in all_profiles:
        if p["file_hash"] == resume_file_hash.strip():
            profile_data = p
            break

    if not profile_data:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found with resume_file_hash='{resume_file_hash}' for client '{client_id}'."
        )

    profile = profile_data["profile"]

    # Generate the document
    output_dir = Path("data/rendered") / client_id.strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename: {candidate_name}_{hash_prefix}.docx
    safe_name = (profile.full_name or "candidate").replace(" ", "_")
    output_filename = f"{safe_name}_{resume_file_hash[:8]}.docx"
    output_path = output_dir / output_filename

    try:
        _render_single(profile, template_path, output_path)
    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    logger.info(f"Generated doc for {profile.full_name} → {output_path}")

    return FileResponse(
        path=str(output_path),
        filename=output_filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-resume-matcher"}
