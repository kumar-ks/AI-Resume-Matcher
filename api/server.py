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
from matching_engine.observability import create_trace, create_span, end_span, flush


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — writes to logs/api.log with daily rollover
# ─────────────────────────────────────────────────────────────────────────────

def _setup_api_logging():
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "api.log"
    print(f"[LOGGING] Writing logs to: {log_file.resolve()}")

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
                    location TEXT,
                    current_company TEXT,
                    current_designation TEXT,
                    total_experience_years REAL,
                    relevant_experience_years REAL,
                    qualification_percentage REAL,
                    match_label TEXT,
                    recommendation TEXT,
                    reasoning TEXT,
                    score_breakdown JSONB DEFAULT '{}',
                    matched_skills JSONB DEFAULT '[]',
                    missing_skills JSONB DEFAULT '[]',
                    top_skills JSONB DEFAULT '[]',
                    jd_match_summary JSONB DEFAULT '{}',
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
    explain: bool = Form(True, description="Run Stage 5 LLM explanations (true=detailed but slow, false=fast rule-based)"),
):
    """
    Upload resumes + JD. Ingests resumes, matches against JD, saves results.
    Returns immediately (202). Processing runs in background.

    Set explain=false for fast results (skips LLM explanations, uses rule-based fallback).
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
        client_id.strip(), job_id.strip(), str(resume_dir), str(jd_path), explain
    )

    return JSONResponse(status_code=202, content={
        "message": f"Ingest started. {len(files)} resumes + JD queued for processing.",
        "client_id": client_id.strip(),
        "job_id": job_id.strip(),
        "files_received": len(files),
    })


async def _background_ingest_and_match(client_id: str, job_id: str, resume_dir: str, jd_path: str, explain: bool = True):
    """Background task: ingest resumes, then match ALL profiles against JD, save results."""
    from matching_engine.database import ProfileDatabase
    from matching_engine.vector_store import VectorStore
    from matching_engine.scanner import scan_and_ingest
    from matching_engine.file_loader import extract_text

    # Create a top-level LangFuse trace for the full ingest+match operation
    trace = create_trace(
        name="api-ingest-and-match",
        client_id=client_id,
        job_id=job_id,
        metadata={"resume_dir": resume_dir, "jd_path": jd_path, "explain": explain},
        tags=["api", "ingest"],
    )

    try:
        # Step 1: Ingest resumes
        ingest_span = create_span(trace, "ingest-resumes", metadata={"resume_dir": resume_dir})

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

        end_span(ingest_span, output={
            "new_count": result["new_count"],
            "skipped_count": result["skipped_count"],
            "failed_count": result["failed_count"],
        })

        db.close()
        vs.close()

        # Step 2: Match ALL profiles against JD
        jd_text = extract_text(Path(jd_path))
        if jd_text and jd_text.strip():
            await _run_match_and_save(client_id, job_id, jd_text, explain=explain, trace_parent=trace)

    except Exception as e:
        logger.error(f"Background ingest+match failed: {e}")
        end_span(create_span(trace, "error"), level="ERROR", status_message=str(e))
    finally:
        flush()


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/match — Upload JD, match against existing profiles, save results
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/match", dependencies=[Depends(verify_api_key)])
async def match_jd(
    background_tasks: BackgroundTasks,
    client_id: str = Form(..., description="Client identifier (NDA isolation)"),
    job_id: str = Form(..., description="Job opening identifier"),
    jd_file: UploadFile = File(..., description="Job Description file (PDF/DOCX/TXT)"),
    explain: bool = Form(True, description="Run Stage 5 LLM explanations (true=detailed but slow, false=fast rule-based)"),
):
    """
    Match JD against already-ingested profiles. Saves results to DB.
    Returns immediately (202). Processing runs in background.

    Set explain=false for fast results (skips LLM explanations, uses rule-based fallback).
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
        _background_match, client_id.strip(), job_id.strip(), str(jd_path), explain
    )

    return JSONResponse(status_code=202, content={
        "message": "Match started. Results will be available via /api/status.",
        "client_id": client_id.strip(),
        "job_id": job_id.strip(),
    })


async def _background_match(client_id: str, job_id: str, jd_path: str, explain: bool = True):
    """Background task: match JD against existing profiles, save results."""
    from matching_engine.file_loader import extract_text

    # Create a top-level LangFuse trace for the match operation
    trace = create_trace(
        name="api-match",
        client_id=client_id,
        job_id=job_id,
        metadata={"jd_path": jd_path, "explain": explain},
        tags=["api", "match"],
    )

    try:
        jd_text = extract_text(Path(jd_path))
        if not jd_text or not jd_text.strip():
            logger.error(f"Could not extract text from JD: {jd_path}")
            end_span(create_span(trace, "error"), level="ERROR", status_message="Empty JD text")
            return

        await _run_match_and_save(client_id, job_id, jd_text, explain=explain, trace_parent=trace)

    except Exception as e:
        logger.error(f"Background match failed: {e}")
        end_span(create_span(trace, "error"), level="ERROR", status_message=str(e))
    finally:
        flush()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: Run matching pipeline and save results to match_results table
# ─────────────────────────────────────────────────────────────────────────────

async def _run_match_and_save(client_id: str, job_id: str, jd_text: str, explain: bool = True, trace_parent=None):
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

    # Create a span for the match+save phase
    match_span = create_span(trace_parent, "match-and-save", metadata={
        "client_id": client_id,
        "job_id": job_id,
        "explain": explain,
    })

    db = ProfileDatabase()
    vs = VectorStore()

    profile_count = db.get_profile_count(client_id)
    if profile_count == 0:
        logger.warning(f"No profiles for client={client_id}, skipping match")
        end_span(match_span, output={"skipped": True, "reason": "no_profiles"})
        db.close()
        vs.close()
        return

    # Stage 1: Extract JD requirements
    model = os.environ.get("MODEL", "ollama/llama3")
    jd_understanding = JDUnderstanding(model=model, temperature=0.1)
    jd = await jd_understanding.extract(jd_text, trace_parent=match_span)
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

    # Stage 5: Explanations (conditional on explain flag)
    explainability = ExplainabilityEngine(model=model, temperature=0.3)

    if explain:
        # Full LLM explanations for top 5 (slower, ~20 sec per candidate)
        logger.info(f"Stage 5: Generating LLM explanations for top 5 (explain=true)")
        for result in results[:5]:
            explanation = await explainability.explain(
                jd, result.candidate, result.scoring_breakdown, result.semantic_scores,
                trace_parent=match_span,
            )
            result.key_strengths = explanation.matched_strengths
            result.missing_skills = explanation.missing_skills
            result.reasoning = explanation.reason_for_score
            result.recommendation = explanation.recommendation

        # Rule-based for the rest
        for result in results[5:]:
            explanation = explainability._fallback_explanation(
                jd, result.candidate, result.scoring_breakdown
            )
            result.key_strengths = explanation.matched_strengths
            result.missing_skills = explanation.missing_skills
            result.reasoning = explanation.reason_for_score
            result.recommendation = explanation.recommendation
    else:
        # Fast mode: rule-based fallback for ALL candidates (no LLM calls)
        logger.info(f"Stage 5: Skipped LLM explanations (explain=false, using rule-based)")
        for result in results:
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

    # JD skills for matching analysis
    jd_must_have = [s.name for s in jd.must_have_skills]
    jd_good_to_have = [s.name for s in jd.good_to_have_skills]
    total_jd_skills = len(jd_must_have) + len(jd_good_to_have)

    for r in results:
        resume_file_hash = hash_lookup.get(r.candidate.raw_text, "")
        candidate = r.candidate

        # ── Build matched skills with proficiency ─────────────────────────
        matched_skills_list = []
        candidate_skills_lower = {s.lower(): s for s in candidate.skills}

        for jd_skill in jd_must_have + jd_good_to_have:
            skill_lower = jd_skill.lower()
            if skill_lower in candidate_skills_lower:
                # Determine proficiency based on experience mentions
                proficiency = _infer_proficiency(jd_skill, candidate)
                matched_skills_list.append({
                    "skill": jd_skill,
                    "proficiency": proficiency,
                    "matched": True,
                })

        # ── Build missing skills with priority ────────────────────────────
        missing_skills_list = []
        for skill in jd_must_have:
            if skill.lower() not in candidate_skills_lower:
                missing_skills_list.append({"skill": skill, "priority": "High"})
        for skill in jd_good_to_have:
            if skill.lower() not in candidate_skills_lower:
                missing_skills_list.append({"skill": skill, "priority": "Low"})

        # ── Match label ───────────────────────────────────────────────────
        pct = r.qualification_percentage
        if pct >= 75:
            match_label = "EXCELLENT MATCH"
        elif pct >= 60:
            match_label = "GOOD MATCH"
        elif pct >= 45:
            match_label = "PARTIAL MATCH"
        else:
            match_label = "WEAK MATCH"

        # ── Current company/designation from latest work experience ───────
        current_company = ""
        current_designation = ""
        if candidate.work_experiences:
            latest = candidate.work_experiences[0]
            current_company = latest.company or ""
            current_designation = latest.title or ""

        # ── Relevant experience (years in matching domain) ────────────────
        relevant_exp = _compute_relevant_experience(candidate, jd)

        # ── JD Match Summary ─────────────────────────────────────────────
        matched_count = len(matched_skills_list)
        partially_matched = len([s for s in candidate.skills
                                 if any(jd_s.lower() in s.lower() or s.lower() in jd_s.lower()
                                        for jd_s in jd_must_have + jd_good_to_have)]) - matched_count
        partially_matched = max(0, partially_matched)
        missing_count = len(missing_skills_list)

        jd_match_summary = {
            "total_jd_skills": total_jd_skills,
            "matched_skills": matched_count,
            "partially_matched": partially_matched,
            "missing_skills": missing_count,
            "min_experience_required": jd.experience_range_min,
            "candidate_experience": candidate.total_experience_years,
            "candidate_relevant_experience": relevant_exp,
        }

        # ── Score breakdown (percentage-based for UI) ─────────────────────
        score_breakdown = {
            "skills_match": round(r.scoring_breakdown.must_have_match * 100, 1),
            "experience_match": round(r.scoring_breakdown.experience_match * 100, 1),
            "skills_depth": round(r.scoring_breakdown.skills_depth * 100, 1),
            "project_relevance": round(r.scoring_breakdown.project_relevance * 100, 1),
            "recency_factor": round(r.scoring_breakdown.recency_factor * 100, 1),
            "overall_fitment": round(r.qualification_percentage, 1),
        }

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO match_results (
                    client_id, job_id, resume_file_hash, full_name, email, phone,
                    location, current_company, current_designation,
                    total_experience_years, relevant_experience_years,
                    qualification_percentage, match_label,
                    recommendation, reasoning,
                    score_breakdown, matched_skills, missing_skills,
                    top_skills, jd_match_summary
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (client_id, job_id, resume_file_hash) DO UPDATE SET
                    qualification_percentage = EXCLUDED.qualification_percentage,
                    match_label = EXCLUDED.match_label,
                    recommendation = EXCLUDED.recommendation,
                    reasoning = EXCLUDED.reasoning,
                    score_breakdown = EXCLUDED.score_breakdown,
                    matched_skills = EXCLUDED.matched_skills,
                    missing_skills = EXCLUDED.missing_skills,
                    top_skills = EXCLUDED.top_skills,
                    jd_match_summary = EXCLUDED.jd_match_summary,
                    location = EXCLUDED.location,
                    current_company = EXCLUDED.current_company,
                    current_designation = EXCLUDED.current_designation,
                    relevant_experience_years = EXCLUDED.relevant_experience_years,
                    matched_at = NOW()
            """, (
                client_id, job_id, resume_file_hash,
                candidate.full_name, candidate.email, candidate.phone,
                candidate.location, current_company, current_designation,
                candidate.total_experience_years, relevant_exp,
                r.qualification_percentage, match_label,
                r.recommendation, r.reasoning,
                json.dumps(score_breakdown),
                json.dumps(matched_skills_list),
                json.dumps(missing_skills_list),
                json.dumps(candidate.skills[:10]),
                json.dumps(jd_match_summary),
            ))
    conn.commit()
    conn.close()
    db.close()
    vs.close()

    end_span(match_span, output={
        "results_count": len(results),
        "top_score": results[0].qualification_percentage if results else 0,
        "profile_count": profile_count,
    })

    logger.info(f"Match complete: {len(results)} results saved for client={client_id}, job={job_id}")


def _infer_proficiency(skill: str, candidate) -> str:
    """Infer proficiency level for a skill based on experience mentions."""
    skill_lower = skill.lower()
    mentions = 0

    # Count how many work experiences mention this skill
    for exp in candidate.work_experiences:
        techs_lower = [t.lower() for t in exp.technologies]
        resps_lower = " ".join(exp.responsibilities).lower()
        if skill_lower in techs_lower or skill_lower in resps_lower:
            mentions += 1

    # Also check projects
    for proj in candidate.projects:
        if skill_lower in [t.lower() for t in proj.technologies]:
            mentions += 1

    if mentions >= 4:
        return "Expert"
    elif mentions >= 2:
        return "Advanced"
    elif mentions >= 1:
        return "Intermediate"
    else:
        return "Beginner"


def _compute_relevant_experience(candidate, jd) -> float:
    """Compute years of experience relevant to the JD domain."""
    jd_domains = set(d.lower() for d in jd.domain_industry)
    jd_skills = set(s.name.lower() for s in jd.must_have_skills)

    relevant_years = 0.0
    for exp in candidate.work_experiences:
        exp_techs = set(t.lower() for t in exp.technologies)
        exp_domain = (exp.domain or "").lower()

        # Check if this experience is relevant
        has_domain_match = exp_domain in jd_domains if exp_domain else False
        has_skill_overlap = bool(exp_techs & jd_skills)

        if has_domain_match or has_skill_overlap:
            if exp.duration_months:
                relevant_years += exp.duration_months / 12.0
            elif exp.start_year and exp.end_year:
                relevant_years += exp.end_year - exp.start_year

    return round(relevant_years, 1)


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
                   location, current_company, current_designation,
                   total_experience_years, relevant_experience_years,
                   qualification_percentage, match_label,
                   recommendation, reasoning,
                   score_breakdown, matched_skills, missing_skills,
                   top_skills, jd_match_summary, matched_at
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
            "summary": {
                "full_name": row["full_name"],
                "email": row["email"],
                "phone": row["phone"],
                "location": row["location"],
                "current_company": row["current_company"],
                "current_designation": row["current_designation"],
                "total_experience_years": row["total_experience_years"],
                "relevant_experience_years": row["relevant_experience_years"],
            },
            "overall_match_score": row["qualification_percentage"],
            "match_label": row["match_label"],
            "score_breakdown": row["score_breakdown"] if isinstance(row["score_breakdown"], dict) else json.loads(row["score_breakdown"] or "{}"),
            "matched_skills": row["matched_skills"] if isinstance(row["matched_skills"], list) else json.loads(row["matched_skills"] or "[]"),
            "missing_skills": row["missing_skills"] if isinstance(row["missing_skills"], list) else json.loads(row["missing_skills"] or "[]"),
            "reasoning": row["reasoning"],
            "recommendation": row["recommendation"],
            "top_skills": row["top_skills"] if isinstance(row["top_skills"], list) else json.loads(row["top_skills"] or "[]"),
            "jd_match_summary": row["jd_match_summary"] if isinstance(row["jd_match_summary"], dict) else json.loads(row["jd_match_summary"] or "{}"),
            "matched_at": row["matched_at"].isoformat() if row["matched_at"] else None,
        })

    # Mark as delivered — REMOVED. UI must explicitly call POST /api/deliver.

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
# POST /api/deliver — Mark results as delivered (called by UI after consuming)
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel as PydanticBaseModel
from typing import List


class DeliverRequest(PydanticBaseModel):
    client_id: str
    job_id: str
    result_ids: List[str]


@app.post("/api/deliver", dependencies=[Depends(verify_api_key)])
async def deliver_results(payload: DeliverRequest):
    """
    Mark results as delivered. Called by UI after successfully consuming results.

    Once marked, these results won't appear in subsequent /api/status calls.
    """
    if not payload.client_id or not payload.client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")
    if not payload.job_id or not payload.job_id.strip():
        raise HTTPException(status_code=400, detail="job_id is required")
    if not payload.result_ids:
        raise HTTPException(status_code=400, detail="result_ids list is required")

    import psycopg
    from psycopg.rows import dict_row

    db_url = os.environ.get("DATABASE_URL", "postgresql://matcher:matcher_secret@localhost:5432/resume_matcher")
    conn = psycopg.connect(db_url, row_factory=dict_row)

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE match_results SET is_delivered = TRUE "
            "WHERE client_id = %s AND job_id = %s AND id = ANY(%s::uuid[])",
            (payload.client_id.strip(), payload.job_id.strip(), payload.result_ids)
        )
        delivered_count = cur.rowcount

    conn.commit()
    conn.close()

    logger.info(f"/api/deliver: {delivered_count} results marked delivered for client={payload.client_id}, job={payload.job_id}")

    return {
        "message": f"{delivered_count} results marked as delivered",
        "client_id": payload.client_id.strip(),
        "job_id": payload.job_id.strip(),
        "delivered_count": delivered_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-resume-matcher"}
