# AI Resume Matcher — Project Steering

## Overview

Python-based AI-powered resume-to-JD matching engine with a 6-stage pipeline. Production-grade system with PostgreSQL + pgvector, multi-tenant client isolation (NDA enforcement), multi-field embeddings, hybrid search (vector + full-text), hallucination detection, scoring evaluation, and a FastAPI REST server for UI integration.

## Architecture

```
┌─────────────────────────┐         REST API          ┌─────────────────────────────┐
│      UI SERVER          │ ──────────────────────────▶│        AI SERVER            │
│                         │                            │                             │
│  - Portal/Dashboard     │  POST /api/ingest          │  - LLM (Ollama/Cloud)       │
│  - 5M resumes in its DB │  POST /api/match           │  - PostgreSQL + pgvector    │
│  - User uploads JDs     │  GET  /api/status          │  - FastAPI server           │
│  - Shows results        │◀──────────────────────────│  - Matching pipeline        │
│                         │         JSON responses     │                             │
└─────────────────────────┘                            └─────────────────────────────┘
```

Two-phase processing: **Ingest** (one-time LLM extraction) and **Match** (fast scoring from DB).

### Pipeline Stages

| Stage | Module | LLM? | Purpose |
|-------|--------|------|---------|
| 1 | `jd_understanding.py` | Yes | Extract structured requirements from JD |
| 2 | `resume_understanding.py` | Yes | Extract structured profile from resume |
| 3 | `semantic_matching.py` | No | Embedding similarity (6 dimensions) |
| 4 | `scoring.py` | No | Weighted formula → qualification % |
| 5 | `explainability.py` | Optional | LLM reasoning for top N |
| 6 | `template_renderer.py` | No | DOCX generation |

### Key Modules

- `run.py` — CLI entry point
- `api/server.py` — FastAPI REST server (production entry point)
- `api/auth.py` — API key authentication
- `api/tasks.py` — Async task manager for batch ingest
- `matching_engine/pipeline.py` — Pipeline orchestrator
- `matching_engine/models.py` — Pydantic data models
- `matching_engine/database.py` — PostgreSQL profile storage (client-scoped)
- `matching_engine/vector_store.py` — pgvector multi-field embeddings + hybrid search
- `matching_engine/scanner.py` — File scanner with TF-IDF + hallucination checks
- `matching_engine/hallucination_check.py` — Grounding verification
- `matching_engine/evaluation.py` — Scoring validation + bias detection

## Technology Stack

- **Language:** Python 3.10+
- **Data models:** Pydantic v2
- **LLM interface:** LiteLLM (Ollama, OpenAI, Anthropic, AWS Bedrock)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)
- **Database:** PostgreSQL 16 + pgvector extension (single container)
- **Vector search:** pgvector HNSW index (cosine similarity)
- **Full-text search:** PostgreSQL tsvector + ts_rank (replaces BM25)
- **Hybrid search:** 65% vector RRF + 35% full-text, fused scoring
- **API server:** FastAPI + uvicorn
- **Auth:** API key via X-API-Key header
- **Container:** Docker (pgvector/pgvector:pg16 image)
- **File parsing:** PyPDF2, pdfplumber, python-docx, textutil/antiword/libreoffice (for .doc)
- **Config:** YAML (PyYAML)
- **Async:** asyncio for concurrent processing
- **Logging:** TimedRotatingFileHandler (daily rollover, 30-day retention)

## Database (PostgreSQL + pgvector)

Single Docker container runs both structured storage and vector search.

### Tables

**resume_profiles** — Structured candidate data (JSONB fields for skills, work history, etc.)
```sql
CREATE TABLE resume_profiles (
    id SERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    source_file TEXT, first_name TEXT, last_name TEXT, email TEXT, phone TEXT,
    skills JSONB, total_experience_years REAL, work_experiences JSONB,
    education JSONB, certifications JSONB, domain_expertise JSONB,
    raw_text TEXT, extracted_at TIMESTAMPTZ,
    UNIQUE(client_id, file_hash)
);
```

**resume_embeddings** — Multi-field vector embeddings (3 rows per resume)
```sql
CREATE TABLE resume_embeddings (
    id SERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    field_type TEXT NOT NULL,  -- 'skills', 'experience', 'summary'
    content TEXT,
    embedding VECTOR(384),
    metadata JSONB,
    UNIQUE(client_id, file_hash, field_type)
);
-- HNSW index for fast ANN search
CREATE INDEX idx_embeddings_hnsw ON resume_embeddings USING hnsw (embedding vector_cosine_ops);
-- GIN index for full-text search
CREATE INDEX idx_embeddings_fts ON resume_embeddings USING gin (to_tsvector('english', content));
```

### Connection

Default: `postgresql://matcher:matcher_secret@localhost:5432/resume_matcher`
Override via `DATABASE_URL` environment variable.

## Multi-Tenant Isolation (NDA Enforcement)

1. Every profile and embedding is tagged with `client_id`
2. `--client-id` and `--job-id` are mandatory for `--ingest` and `--match`
3. All SQL queries include `WHERE client_id = %s`
4. Vector search includes `WHERE client_id = %s` filter
5. Within same client, resumes are shared across job_ids freely
6. Violation = NDA breach — never bypass client_id checks

## API Server

### Authentication

API key passed via `X-API-Key` header. Keys set via `AI_MATCHER_API_KEYS` env var (comma-separated).

Generate a key: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

### Endpoints

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| `POST` | `/api/ingest` | Yes | Upload resumes + JD, ingest, match, save results (async) |
| `POST` | `/api/match` | Yes | Upload JD, match existing profiles, save results (async) |
| `GET` | `/api/status` | Yes | Get unsent results for client_id + job_id |
| `POST` | `/api/template` | Yes | Upload DOCX template for a client (latest always wins) |
| `POST` | `/api/generate-doc` | Yes | Convert candidate resume into client's template (returns DOCX file) |
| `GET` | `/health` | No | Health check |

### Ingest Flow (API)

```
UI sends POST /api/ingest (resumes + JD + client_id + job_id)
    → Files saved to data/uploads/{client_id}/{job_id}/
    → Returns 202 immediately
    → Background: ingests resumes → matches against JD → saves to match_results table
    → UI calls GET /api/status?client_id=X&job_id=Y to fetch results
```

### Match Flow (API)

```
UI sends POST /api/match (JD + client_id + job_id)
    → Returns 202 immediately
    → Background: matches existing profiles against JD → saves to match_results table
    → UI calls GET /api/status to fetch results
```

### Status Response Structure

```json
{
  "client_id": "ACME_CORP",
  "job_id": "JOB-001",
  "total_results": 5,
  "results": [
    {
      "result_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "resume_file_hash": "d41d8cd98f00b204e9800998ecf8427e",
      "full_name": "Kumar S Karpuram",
      "email": "shootmail2kumar@gmail.com",
      "phone": "+91-96864-88688",
      "total_experience_years": 17.0,
      "qualification_percentage": 59.7,
      "recommendation": "Consider for interview",
      "reasoning": "Strong match on MLOps, Docker, Kubernetes...",
      "key_strengths": ["MLOps", "Docker", "Kubernetes"],
      "missing_skills": ["Scala"],
      "top_skills": ["Python", "AWS", "Terraform", "Docker", "Kubernetes"],
      "scoring_breakdown": {
        "must_have_match": 0.82,
        "experience_match": 0.75,
        "skills_depth": 0.68,
        "project_relevance": 0.45,
        "recency_factor": 0.90
      },
      "matched_at": "2026-07-12T23:30:48Z"
    }
  ]
}
```

Results are marked as delivered after being returned — subsequent calls only return new/updated results.

## Embedding Strategy (Multi-Field)

Three rows per resume in `resume_embeddings`:
- `skills` — Technologies, tools, certifications (weight: 0.45)
- `experience` — Role titles, companies, domains (weight: 0.35)
- `summary` — Career summary, achievements (weight: 0.20)

TF-IDF keyword extraction removes generic filler before embedding.

## Hybrid Search

At query time:
1. JD embedded → cosine similarity search per field type (pgvector HNSW)
2. Full-text search via PostgreSQL `ts_rank` + `plainto_tsquery`
3. Fusion: 65% vector (Reciprocal Rank Fusion across 3 fields) + 35% full-text
4. Top-N candidates returned for scoring

## Hallucination Detection

After LLM extraction, verifies against source text:
- Skills: compound strings split, >50% sub-terms must be found
- Companies: lenient matching with suffix stripping
- Certifications: fuzzy match with tech aliases
- Experience years: cross-check vs work history dates
- Name: verify in source text

## Evaluation Framework

After matching:
- Ranking stability (weight perturbation ±10%)
- Score distribution analysis
- Calibration bands (Strong/Good/Partial/Weak)
- Bias detection (experience/skill-count correlation)
- Keyword stuffing detection

## Logging

- `logs/ingest.log` and `logs/match.log`
- Daily rollover at midnight, 30-day retention
- All output (logger + print) captured
- Format: `2026-07-12 23:30:48 | INFO | module | message`

## File Organization

```
AI-Resume-Matcher/
├── run.py                    ← CLI entry point
├── docker-compose.yml        ← PostgreSQL + pgvector container
├── config.yaml               ← Runtime configuration
├── requirements.txt          ← Dependencies
├── .env                      ← API keys, DATABASE_URL (git-ignored)
├── api/
│   ├── __init__.py
│   ├── server.py             ← FastAPI endpoints
│   ├── auth.py               ← API key authentication
│   └── tasks.py              ← Async task manager
├── matching_engine/
│   ├── models.py             ← Pydantic data models
│   ├── database.py           ← PostgreSQL profile storage
│   ├── vector_store.py       ← pgvector + hybrid search
│   ├── scanner.py            ← Ingest with TF-IDF + hallucination
│   ├── hallucination_check.py
│   ├── evaluation.py
│   ├── jd_understanding.py   ← Stage 1
│   ├── resume_understanding.py ← Stage 2
│   ├── semantic_matching.py  ← Stage 3
│   ├── scoring.py            ← Stage 4
│   ├── explainability.py     ← Stage 5
│   ├── template_renderer.py  ← Stage 6
│   ├── pipeline.py
│   ├── llm_client.py
│   ├── file_loader.py
│   └── utils.py
├── logs/                     ← Daily rolling logs
├── data/uploads/             ← Files received via API
├── resumes/                  ← CLI input resumes (git-ignored)
├── jd/                       ← CLI input JDs (git-ignored)
└── template/                 ← DOCX template (read-only)
```

## Build & Run

### Prerequisites

```bash
# Start PostgreSQL + pgvector
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt
```

### CLI Mode (dev/testing)

```bash
python run.py --ingest --client-id ACME_CORP --job-id JOB-001
python run.py --match --client-id ACME_CORP --job-id JOB-001
python run.py --db-status
```

### API Mode (production)

```bash
# Set API key
export AI_MATCHER_API_KEYS="your-secure-key-here"

# Start server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Swagger docs at http://localhost:8000/docs
```

### Verify Data

```bash
# Connect to PostgreSQL
docker exec -it resume_matcher_db psql -U matcher -d resume_matcher

# Check profiles
SELECT client_id, job_id, COUNT(*) FROM resume_profiles GROUP BY client_id, job_id;

# Check embeddings
SELECT field_type, COUNT(*) FROM resume_embeddings GROUP BY field_type;

# Test vector search
SELECT file_hash, 1-(embedding <=> (SELECT embedding FROM resume_embeddings LIMIT 1)) as sim
FROM resume_embeddings WHERE field_type='skills' ORDER BY sim DESC LIMIT 5;
```

## Supported File Formats

| Format | Handler | Notes |
|--------|---------|-------|
| `.pdf` | PyPDF2 + pdfplumber | Falls back to OCR if text extraction fails |
| `.docx` | python-docx | Extracts paragraphs, tables, text boxes, hyperlinks |
| `.doc` | textutil (macOS) / antiword / libreoffice (Ubuntu) | Legacy binary Word format |
| `.txt` | Direct read | UTF-8 with fallback encodings |

For Ubuntu production: `sudo apt install antiword` (lightweight) or `sudo apt install libreoffice` (full).

## API Result Delivery Logic

- `/api/ingest` and `/api/match` save results to `match_results` table with `is_delivered = FALSE`
- `/api/status` returns only `is_delivered = FALSE` results, then marks them `is_delivered = TRUE`
- On re-ingest with new resumes: only NEW resume results get `is_delivered = FALSE`
- Previously delivered results for existing resumes are NOT reset unless their score changes

## Important Constraints

- **NDA isolation:** `--client-id` mandatory. Never bypass.
- **API auth:** All endpoints (except /health) require X-API-Key header
- **PII:** Never commit resumes, JDs, or `.env` to git
- **Scoring weights** must sum to 1.0
- **Ollama auto-management:** Script auto-starts and auto-pulls models
- **SSL/proxy:** Patches SSL for corporate environments (Zscaler)
- **Database:** Always use parameterized queries (psycopg handles this)

## Coding Standards

- Type hints on all function signatures
- Pydantic BaseModel for all structured data
- async/await for I/O-bound operations
- `logging` module for internal state; `print()` for CLI output
- Config priority: CLI flags > env vars > config.yaml > defaults
- All DB reads require client_id parameter (enforced with ValueError)
