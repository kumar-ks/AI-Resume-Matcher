"""
FastAPI Server — AI Resume Matcher API
========================================

Exposes the matching engine as a REST API for the UI server to call.

ENDPOINTS:
    POST /api/ingest         — Upload batch of resumes for async processing
    GET  /api/ingest/{id}    — Poll ingest task status
    POST /api/match          — Match JD against stored profiles (client-scoped)
    GET  /api/status         — DB stats for a client

AUTH:
    All endpoints require X-API-Key header.
    Set API keys via AI_MATCHER_API_KEYS env var (comma-separated).

USAGE:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from api.auth import verify_api_key
from api.tasks import task_manager, TaskStatus

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Matcher API",
    description="Multi-tenant resume matching engine with NDA-level client isolation",
    version="2.0.0",
)

# Upload directory for received resumes
UPLOAD_BASE = Path("data/uploads")


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/ingest — Batch upload resumes for async processing
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_resumes(
    background_tasks: BackgroundTasks,
    client_id: str = Form(..., description="Client identifier (NDA isolation)"),
    job_id: str = Form(..., description="Job opening identifier"),
    files: list[UploadFile] = File(..., description="Resume files (PDF/DOCX/TXT)"),
):
    """
    Upload a batch of resumes for async ingestion.

    Returns a task_id immediately. Poll /api/ingest/{task_id} for status.

    Files are saved to disk, then processed asynchronously via the LLM pipeline.
    """
    if not client_id or not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")
    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="job_id is required")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    # Save uploaded files to disk
    upload_dir = UPLOAD_BASE / client_id.strip() / job_id.strip()
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file in files:
        # Sanitize filename
        filename = file.filename or "unknown_resume"
        dest = upload_dir / filename

        # Avoid overwriting — add suffix if exists
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = upload_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        # Save file
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved_paths.append(str(dest))

    logger.info(f"Received {len(saved_paths)} files for client={client_id}, job={job_id}")

    # Create async task
    task = task_manager.create_task(
        client_id=client_id.strip(),
        job_id=job_id.strip(),
        file_paths=saved_paths,
    )

    # Run ingest in background
    background_tasks.add_task(task_manager.run_ingest_task, task)

    return JSONResponse(
        status_code=202,
        content={
            "message": f"Ingest task created. {len(files)} files queued for processing.",
            "task_id": task.task_id,
            "client_id": client_id.strip(),
            "job_id": job_id.strip(),
            "files_received": len(files),
            "poll_url": f"/api/ingest/{task.task_id}",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/ingest/{task_id} — Poll ingest task status
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ingest/{task_id}", dependencies=[Depends(verify_api_key)])
async def get_ingest_status(task_id: str):
    """
    Poll the status of an ingest task.

    Returns current progress (files processed, failed, skipped).
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return task.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/match — Match JD against stored profiles
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/match", dependencies=[Depends(verify_api_key)])
async def match_jd(
    client_id: str = Form(..., description="Client identifier (NDA isolation)"),
    job_id: str = Form(..., description="Job opening identifier"),
    jd_file: UploadFile = File(..., description="Job Description file (PDF/DOCX/TXT)"),
    top_n: Optional[int] = Form(None, description="Return only top N candidates"),
):
    """
    Match a JD against stored profiles for a client.

    NDA ENFORCEMENT: Only profiles belonging to the given client_id are considered.
    Returns ranked candidates with scores, strengths, and gaps.
    """
    if not client_id or not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")
    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="job_id is required")

    from matching_engine.database import ProfileDatabase
    from matching_engine.vector_store import VectorStore
    from matching_engine.file_loader import extract_text_from_bytes
    from matching_engine.jd_understanding import JDUnderstanding
    from matching_engine.scoring import Scorer
    from matching_engine.semantic_matching import SemanticMatcher
    from matching_engine.explainability import ExplainabilityEngine
    from matching_engine.models import MatchResult, ExplainabilityReport

    # Extract JD text from uploaded file
    jd_content = await jd_file.read()
    jd_filename = jd_file.filename or "jd.txt"
    
    # Save temporarily to extract text
    tmp_path = Path("data/tmp") / jd_filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(jd_content)

    try:
        from matching_engine.file_loader import extract_text
        jd_text = extract_text(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not jd_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from JD file")

    # Load profiles from DB (client-scoped)
    db = ProfileDatabase()
    vs = VectorStore()

    profile_count = db.get_profile_count(client_id.strip())
    if profile_count == 0:
        db.close()
        raise HTTPException(
            status_code=404,
            detail=f"No profiles found for client '{client_id}'. Run ingest first."
        )

    # Stage 1: Extract JD requirements
    jd_understanding = JDUnderstanding(model="ollama/llama3", temperature=0.1)
    jd = await jd_understanding.extract(jd_text)
    jd.client_id = client_id.strip()
    jd.job_id = job_id.strip()

    # Query vector store (client-scoped)
    top_n_query = min(50, profile_count)
    similar_results = vs.query_similar(jd_text, client_id=client_id.strip(), top_n=top_n_query)

    if similar_results:
        top_hashes = {r["file_hash"] for r in similar_results}
        all_profiles_meta = db.get_all_profiles_with_metadata(client_id.strip())
        candidates = [p for p in all_profiles_meta if p["file_hash"] in top_hashes]
    else:
        candidates = db.get_all_profiles_with_metadata(client_id.strip())

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
            key_strengths=[],
            missing_skills=[],
            reasoning="",
            recommendation="",
        ))

    results.sort(key=lambda r: r.qualification_percentage, reverse=True)

    # Stage 5: Explain top candidates
    explain_count = min(5, len(results))
    explainability = ExplainabilityEngine(model="ollama/llama3", temperature=0.3)

    for result in results[:explain_count]:
        explanation = await explainability.explain(
            jd, result.candidate, result.scoring_breakdown, result.semantic_scores
        )
        result.key_strengths = explanation.matched_strengths
        result.missing_skills = explanation.missing_skills
        result.reasoning = explanation.reason_for_score
        result.recommendation = explanation.recommendation

    for result in results[explain_count:]:
        explanation = explainability._fallback_explanation(
            jd, result.candidate, result.scoring_breakdown
        )
        result.key_strengths = explanation.matched_strengths
        result.missing_skills = explanation.missing_skills
        result.reasoning = explanation.reason_for_score
        result.recommendation = explanation.recommendation

    # Apply top_n filter
    if top_n:
        results = results[:top_n]

    db.close()

    # Build response
    response_candidates = []
    for rank, r in enumerate(results, 1):
        response_candidates.append({
            "rank": rank,
            "full_name": r.candidate.full_name or None,
            "first_name": r.candidate.first_name or None,
            "last_name": r.candidate.last_name or None,
            "email": r.candidate.email or None,
            "phone": r.candidate.phone or None,
            "total_experience_years": r.candidate.total_experience_years,
            "qualification_percentage": r.qualification_percentage,
            "recommendation": r.recommendation,
            "reasoning": r.reasoning,
            "key_strengths": r.key_strengths,
            "missing_skills": r.missing_skills,
            "top_skills": r.candidate.skills[:10],
            "scoring_breakdown": {
                "must_have_match": round(r.scoring_breakdown.must_have_match, 3),
                "experience_match": round(r.scoring_breakdown.experience_match, 3),
                "skills_depth": round(r.scoring_breakdown.skills_depth, 3),
                "project_relevance": round(r.scoring_breakdown.project_relevance, 3),
                "recency_factor": round(r.scoring_breakdown.recency_factor, 3),
            },
        })

    return {
        "client_id": client_id.strip(),
        "job_id": job_id.strip(),
        "jd_title": jd.title,
        "total_candidates_scored": len(results),
        "candidates": response_candidates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/status — Database status for a client
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status", dependencies=[Depends(verify_api_key)])
async def get_status(client_id: Optional[str] = None):
    """
    Get database and vector store status.

    If client_id provided, shows stats for that client only.
    Otherwise shows global stats.
    """
    from matching_engine.database import ProfileDatabase
    from matching_engine.vector_store import VectorStore

    db = ProfileDatabase()
    vs = VectorStore()

    db_status = db.get_status()
    vs_status = vs.get_status()

    response = {
        "database": db_status,
        "vector_store": vs_status,
    }

    if client_id:
        response["client_profiles"] = db.get_profile_count(client_id.strip())

    # Active tasks
    if client_id:
        tasks = task_manager.get_tasks_for_client(client_id.strip())
        response["recent_tasks"] = [t.to_dict() for t in tasks[-10:]]

    db.close()
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Health check (no auth required)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy", "service": "ai-resume-matcher"}
