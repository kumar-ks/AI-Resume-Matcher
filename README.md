# AI Resume Matcher

AI-powered Resume to Job Description matching engine with a 6-stage pipeline. Production-grade system with **PostgreSQL + pgvector**, **multi-tenant client isolation (NDA)**, **hybrid search**, **hallucination detection**, and a **FastAPI REST API** for UI server integration.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Server](#api-server)
- [Multi-Tenant Client Isolation (NDA)](#multi-tenant-client-isolation-nda)
- [Database (PostgreSQL + pgvector)](#database-postgresql--pgvector)
- [Hybrid Search](#hybrid-search)
- [Hallucination Detection](#hallucination-detection)
- [Scoring & Evaluation](#scoring--evaluation)
- [CLI Options](#cli-options)
- [Configuration](#configuration)
- [Logging](#logging)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Quick Start

### 1. Start PostgreSQL + pgvector

```bash
docker-compose up -d
```

This starts a single container with PostgreSQL 16 + pgvector extension.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (for local LLM)

```bash
brew install ollama
ollama pull llama3
```

> The script auto-starts Ollama if not running and auto-pulls the model.

### 4. Place your files

- Drop resumes into the `resumes/` folder (PDF, DOCX, TXT)
- Place the job description into the `jd/` folder (PDF, DOCX, TXT)

### 5. Run (CLI)

```bash
# Ingest resumes into PostgreSQL (requires --client-id and --job-id)
python run.py --ingest --client-id ACME_CORP --job-id JOB-001

# Match against a JD (only ACME_CORP resumes visible)
python run.py --match --client-id ACME_CORP --job-id JOB-001

# Check DB status
python run.py --db-status
```

### 6. Run (API server)

```bash
# Set API key
export AI_MATCHER_API_KEYS="your-secure-key"

# Start server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Swagger docs at http://localhost:8000/docs
```

---

## Architecture

```
┌─────────────────────────┐         REST API          ┌─────────────────────────────┐
│      UI SERVER          │ ──────────────────────────▶│        AI SERVER            │
│                         │                            │                             │
│  - Portal/Dashboard     │  POST /api/ingest          │  - Ollama LLM               │
│  - 5M resumes in its DB │  POST /api/match           │  - PostgreSQL + pgvector    │
│  - User uploads JDs     │  GET  /api/status          │  - FastAPI server           │
│  - Shows results        │◀──────────────────────────│  - Matching pipeline        │
│                         │         JSON responses     │                             │
└─────────────────────────┘                            └─────────────────────────────┘
```

### Pipeline Stages

| Stage | Module | LLM? | Description |
|-------|--------|:----:|-------------|
| 1 | `jd_understanding.py` | Yes (1 call) | Extracts structured requirements from JD |
| 2 | `resume_understanding.py` | Yes (per resume) | Extracts structured profile (ingest only) |
| 3 | `semantic_matching.py` | No | Embedding similarity across 6 dimensions |
| 4 | `scoring.py` | No | Weighted formula → qualification percentage |
| 5 | `explainability.py` | Optional | LLM reasoning for top N candidates |
| 6 | `template_renderer.py` | No | DOCX generation (if `--generate-doc`) |

---

## API Server

### Authentication

All endpoints (except `/health`) require an `X-API-Key` header.

```bash
# Generate a secure key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Set it (comma-separated for multiple keys)
export AI_MATCHER_API_KEYS="xK9m2Fq_7vLpR3nYwT1bHsZcUj8dAeGi0kMoXhNa4E"

# Start server
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Default dev key: `dev-key-change-me` (used if no env var set).

**Test authentication:**

```bash
# ✅ Valid key — returns 200
curl -s http://localhost:8000/api/status \
  -H "X-API-Key: xK9m2Fq_7vLpR3nYwT1bHsZcUj8dAeGi0kMoXhNa4E"

# ❌ Wrong key — returns 403 Forbidden
curl -s http://localhost:8000/api/status \
  -H "X-API-Key: wrong-key-here"

# ❌ No key — returns 401 Unauthorized
curl -s http://localhost:8000/api/status
```

### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/ingest` | Upload resumes + JD, ingest and match (async) |
| `POST` | `/api/match` | Upload JD, match existing profiles (async) |
| `GET` | `/api/status` | Get unsent results for client_id + job_id |
| `POST` | `/api/template` | Upload DOCX template for a client (latest always wins) |
| `POST` | `/api/generate-doc` | Convert candidate resume into client's template (returns DOCX) |
| `GET` | `/health` | Health check (no auth) |

### Example API Calls

```bash
# ─────────────────────────────────────────────────────────────────────────────
# 1. HEALTH CHECK (no auth required)
# ─────────────────────────────────────────────────────────────────────────────
curl http://localhost:8000/health

# Response:
# {"status": "healthy", "service": "ai-resume-matcher"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. INGEST — Upload resumes + JD, ingest and match (async, returns 202)
# ─────────────────────────────────────────────────────────────────────────────
curl -X POST http://localhost:8000/api/ingest \
  -H "X-API-Key: dev-key-change-me" \
  -F "client_id=ACME_CORP" \
  -F "job_id=JOB-001" \
  -F "jd_file=@jd/Forward Deployed Engineer.docx" \
  -F "files=@resumes/Kumar_DevSecOps_MLOps_v1.pdf" \
  -F "files=@resumes/DevSecOps_MLOps_v3.docx" \
  -F "files=@resumes/Arun Prasad Resume.pdf"

# Response (202 Accepted):
# {
#   "message": "Ingest started. 3 resumes + JD queued for processing.",
#   "client_id": "ACME_CORP",
#   "job_id": "JOB-001",
#   "files_received": 3
# }


# ─────────────────────────────────────────────────────────────────────────────
# 3. MATCH — Upload JD only, match against existing profiles (async, returns 202)
# ─────────────────────────────────────────────────────────────────────────────
curl -X POST http://localhost:8000/api/match \
  -H "X-API-Key: dev-key-change-me" \
  -F "client_id=ACME_CORP" \
  -F "job_id=JOB-002" \
  -F "jd_file=@jd/Lead MLOps Engineer.pdf"

# Response (202 Accepted):
# {
#   "message": "Match started. Results will be available via /api/status.",
#   "client_id": "ACME_CORP",
#   "job_id": "JOB-002"
# }


# ─────────────────────────────────────────────────────────────────────────────
# 4. STATUS — Get unsent results for a client + job (marks as delivered)
# ─────────────────────────────────────────────────────────────────────────────
curl "http://localhost:8000/api/status?client_id=ACME_CORP&job_id=JOB-001" \
  -H "X-API-Key: dev-key-change-me"

# Response:
# {
#   "client_id": "ACME_CORP",
#   "job_id": "JOB-001",
#   "total_results": 3,
#   "results": [
#     {
#       "result_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
#       "resume_file_hash": "d41d8cd98f00b204e9800998ecf8427e",
#       "full_name": "Kumar S Karpuram",
#       "email": "shootmail2kumar@gmail.com",
#       "phone": "+91-96864-88688",
#       "total_experience_years": 17.0,
#       "qualification_percentage": 59.7,
#       "recommendation": "Consider for interview",
#       "reasoning": "Strong DevOps/MLOps match...",
#       "key_strengths": ["MLOps", "Docker", "Kubernetes"],
#       "missing_skills": ["Scala"],
#       "top_skills": ["Python", "AWS", "Terraform", "Docker", "Kubernetes"],
#       "scoring_breakdown": {
#         "must_have_match": 0.82,
#         "experience_match": 0.75,
#         "skills_depth": 0.68,
#         "project_relevance": 0.45,
#         "recency_factor": 0.90
#       },
#       "matched_at": "2026-07-19T21:07:04Z"
#     }
#   ]
# }
# NOTE: Calling /api/status again returns empty results (already delivered).


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEMPLATE — Upload a DOCX template for a client (latest always wins)
# ─────────────────────────────────────────────────────────────────────────────
curl -X POST http://localhost:8000/api/template \
  -H "X-API-Key: dev-key-change-me" \
  -F "client_id=ACME_CORP" \
  -F "template_file=@template/Company_Resume_Template.docx"

# Response:
# {
#   "message": "Template uploaded successfully",
#   "client_id": "ACME_CORP",
#   "template_file": "Company_Resume_Template.docx"
# }


# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERATE-DOC — Convert candidate's resume into the client's template
# ─────────────────────────────────────────────────────────────────────────────
curl -X POST http://localhost:8000/api/generate-doc \
  -H "X-API-Key: dev-key-change-me" \
  -F "client_id=ACME_CORP" \
  -F "resume_file_hash=d41d8cd98f00b204e9800998ecf8427e" \
  --output Kumar_S_Karpuram_d41d8cd9.docx

# Response: Binary DOCX file downloaded
```

### API Parameter Reference

#### Request Parameters

| Param | Used in | Type | Description |
|-------|---------|------|-------------|
| `client_id` | All endpoints | string (required) | Client identifier. Enforces NDA isolation — resumes from one client are never visible to another. |
| `job_id` | `/api/ingest`, `/api/match`, `/api/status` | string (required) | Job opening identifier. Tracks which job the resumes/results belong to. Resumes within same client are shared across job_ids. |
| `jd_file` | `/api/ingest`, `/api/match` | file (required) | Job Description document (PDF/DOCX/TXT). Used to extract requirements and match against profiles. |
| `files` | `/api/ingest` | file[] (required) | One or more resume files (PDF/DOCX/TXT). Each `-F "files=@path"` adds one resume. |
| `template_file` | `/api/template` | file (required) | DOCX template file. Latest upload replaces previous. Used by `/api/generate-doc` to format candidate resumes. |
| `resume_file_hash` | `/api/generate-doc` | string (required) | MD5 hash of the candidate's resume file. Identifies which profile to render into the template. Get this from `/api/status` response. |
| `X-API-Key` | All (except /health) | header (required) | API authentication key. Set via `AI_MATCHER_API_KEYS` env var on server. |

#### Response Fields (`/api/status`)

| Field | Type | Description |
|-------|------|-------------|
| `result_id` | UUID | Unique identifier for this specific match result. Auto-generated per (client_id + job_id + resume) combination. Use this to reference a specific match outcome. |
| `resume_file_hash` | string (MD5) | MD5 hash of the resume file content. Same file always produces the same hash. **This is the bridge between UI and AI server** — UI computes the same MD5 on its copy to link results back to the original resume. |
| `full_name` | string | Candidate's extracted full name. |
| `email` | string | Candidate's email address. |
| `phone` | string | Candidate's phone number. |
| `total_experience_years` | float | Total years of professional experience (extracted by LLM). |
| `qualification_percentage` | float (0-100) | Overall match score against the JD. Higher = better fit. |
| `recommendation` | string | Action recommendation (e.g., "Consider for interview", "May need additional screening"). |
| `reasoning` | string | AI-generated explanation of why the candidate scored this way. |
| `key_strengths` | string[] | Skills/qualities that matched the JD well. |
| `missing_skills` | string[] | JD requirements the candidate lacks. |
| `top_skills` | string[] | Candidate's top 10 extracted skills. |
| `scoring_breakdown` | object | Per-dimension scores (must_have_match, experience_match, skills_depth, project_relevance, recency_factor). Each 0.0–1.0. |
| `matched_at` | ISO timestamp | When this match was computed. |

#### How `resume_file_hash` bridges UI and AI Server

```
UI Server:                              AI Server:
1. User uploads resume.pdf              
2. UI computes MD5 → "abc123"           
3. UI stores resume + hash in its DB    
4. UI sends file to /api/ingest         → AI computes same MD5 → "abc123"
5. UI calls /api/status                 → AI returns resume_file_hash: "abc123"
6. UI matches hash to its own record    → LINKED. Same file. No extra API call.
```

---

## Multi-Tenant Client Isolation (NDA)

Strict client-level data isolation. Resumes for one client can **never** be accessed by another.

| Rule | How |
|------|-----|
| Resumes for Client A invisible to Client B | All SQL queries filter by `client_id` |
| `--client-id` and `--job-id` mandatory | CLI validation exits if missing |
| Same file can exist under different clients | `UNIQUE(client_id, resume_file_hash)` constraint in match_results |
| Within same client, resumes shared across jobs | `job_id` for tracking, not isolation |

```bash
# Client A
python run.py --ingest --client-id CLIENT_A --job-id JOB-101
python run.py --match --client-id CLIENT_A --job-id JOB-102

# Client B (completely separate, no cross-visibility)
python run.py --ingest --client-id CLIENT_B --job-id JOB-201
python run.py --match --client-id CLIENT_B --job-id JOB-201
```

---

## Database (PostgreSQL + pgvector)

Single Docker container (`pgvector/pgvector:pg16`) provides both structured storage and vector search.

```bash
# Start
docker-compose up -d

# Connect
docker exec -it resume_matcher_db psql -U matcher -d resume_matcher
```

### Tables

| Table | Purpose |
|-------|---------|
| `resume_profiles` | Structured candidate data (JSONB for skills, work history) |
| `resume_embeddings` | Multi-field vector embeddings (3 rows per resume) |

### Verify Data

```sql
-- Profile count per client
SELECT client_id, job_id, COUNT(*) FROM resume_profiles GROUP BY client_id, job_id;

-- Embeddings per field type
SELECT field_type, COUNT(*) FROM resume_embeddings GROUP BY field_type;

-- Test vector similarity
SELECT file_hash, metadata->>'full_name',
       1-(embedding <=> (SELECT embedding FROM resume_embeddings WHERE field_type='skills' LIMIT 1)) as similarity
FROM resume_embeddings WHERE field_type='skills'
ORDER BY similarity DESC LIMIT 5;

-- Full-text search
SELECT file_hash, metadata->>'full_name',
       ts_rank(to_tsvector('english', content), plainto_tsquery('english', 'kubernetes docker')) as rank
FROM resume_embeddings
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'kubernetes docker')
ORDER BY rank DESC;
```

---

## Hybrid Search

Combines two search methods for better retrieval:

| Method | Weight | How |
|--------|--------|-----|
| Vector search (semantic) | 65% | pgvector HNSW cosine similarity across 3 field types, fused with RRF |
| Full-text search (lexical) | 35% | PostgreSQL `ts_rank` + `tsvector` (exact keyword matching) |

Multi-field embeddings per resume:
- `skills` (weight 0.45) — Technologies, tools, certifications
- `experience` (weight 0.35) — Role titles, companies, domains
- `summary` (weight 0.20) — Career summary, achievements

TF-IDF keyword extraction removes generic filler words before embedding.

---

## Hallucination Detection

During ingest, each LLM extraction is verified against the source resume text:

- **Skills:** Compound strings split on delimiters; >50% of sub-terms must be found
- **Companies:** Lenient matching with suffix stripping (Inc, Ltd, Corp)
- **Certifications:** Fuzzy match with tech aliases (k8s ↔ Kubernetes)
- **Experience years:** Cross-checks claimed total vs work history date span
- **Name:** Verifies extracted name parts appear in source text

Reports confidence score (0–100%). Flags unreliable extractions with warnings.

---

## Scoring & Evaluation

### Scoring Weights

| Dimension | Weight | Question |
|-----------|--------|----------|
| `must_have_match` | 0.35 | Does the candidate have required skills? |
| `experience_match` | 0.25 | Right level of seniority? |
| `skills_depth` | 0.20 | Deep expertise or just keywords? |
| `project_relevance` | 0.12 | Practical evidence of applied skills? |
| `recency_factor` | 0.08 | Is experience current? |

Weights must sum to 1.0. Configurable in `config.yaml`.

### Evaluation (runs after every match)

- **Ranking stability:** Perturbs weights ±10%, checks if top-3 changes
- **Score distribution:** Detects broken pipelines (all-same, too-narrow)
- **Bias detection:** Flags experience over-weighting, skill-count correlation
- **Keyword stuffing:** High keyword match + low depth = suspect

---

## CLI Options

| Flag | Description |
|------|-------------|
| `--client-id` | Client identifier (REQUIRED for ingest/match) |
| `--job-id` | Job opening identifier (REQUIRED for ingest/match) |
| `--ingest` | Ingest resumes into DB |
| `--match` | Match stored profiles against JD |
| `--config` | Path to config file (default: `config.yaml`) |
| `--resumes` | Resume folder path (default: `./resumes`) |
| `--jd` | JD file path |
| `--model` | LLM model (default: from config, e.g. `ollama/llama3`) |
| `--top` | Show only top N candidates |
| `--output` | Save JSON results to file |
| `--debug` | Verbose logging |
| `--concurrency` | Parallel processing count |
| `--explain-top` | Only explain top N (saves LLM calls) |
| `--generate-doc` | Generate DOCX for top N |
| `--scan-mode` | `db_first` / `folder_only` / `db_only` |
| `--db-status` | Show DB stats and exit |

---

## Configuration

`config.yaml`:

```yaml
# LLM
model: "ollama/llama3"
failover_model: null

# Embedding model (sentence-transformers)
embedding_model: "all-MiniLM-L6-v2"

# File paths
resumes_dir: "./resumes"
jd_dir: "./jd"

# Performance
concurrency: 3
explain_top: null
temperature: 0.1
max_tokens: 4096

# Scoring weights (must sum to 1.0)
scoring_weights:
  must_have_match: 0.35
  experience_match: 0.25
  skills_depth: 0.20
  project_relevance: 0.12
  recency_factor: 0.08

# Scan mode: db_first | folder_only | db_only
scan_mode: "db_first"
```

Database connection via `DATABASE_URL` env var (default: `postgresql://matcher:matcher_secret@localhost:5432/resume_matcher`).

---

## Logging

```
logs/
├── ingest.log              ← Written during --ingest
├── ingest.log.2026-07-11   ← Yesterday's rolled log
├── match.log               ← Written during --match
└── match.log.2026-07-11    ← Rolled daily, 30-day retention
```

Format: `2026-07-12 23:30:48 | INFO | module_name | message`

All console output (logger + print) captured in log files.

---

## Project Structure

```
AI-Resume-Matcher/
├── run.py                        ← CLI entry point
├── docker-compose.yml            ← PostgreSQL + pgvector
├── config.yaml                   ← Runtime configuration
├── requirements.txt              ← Python dependencies
├── .env                          ← API keys, DATABASE_URL (git-ignored)
├── .gitignore
├── api/
│   ├── __init__.py
│   ├── server.py                 ← FastAPI endpoints
│   ├── auth.py                   ← API key authentication
│   └── tasks.py                  ← Async task manager
├── matching_engine/
│   ├── models.py                 ← Pydantic data models
│   ├── database.py               ← PostgreSQL profile storage
│   ├── vector_store.py           ← pgvector + hybrid search
│   ├── scanner.py                ← Ingest (TF-IDF + hallucination check)
│   ├── hallucination_check.py    ← Grounding verification
│   ├── evaluation.py             ← Scoring validation + bias detection
│   ├── jd_understanding.py       ← Stage 1: JD parsing
│   ├── resume_understanding.py   ← Stage 2: Resume parsing
│   ├── semantic_matching.py      ← Stage 3: Embedding similarity
│   ├── scoring.py                ← Stage 4: Weighted scoring
│   ├── explainability.py         ← Stage 5: AI reasoning
│   ├── template_renderer.py      ← Stage 6: DOCX generation
│   ├── pipeline.py               ← Pipeline orchestrator
│   ├── llm_client.py             ← LiteLLM wrapper + failover
│   ├── file_loader.py            ← PDF/DOCX/TXT extraction
│   └── utils.py                  ← Shared utilities
├── logs/                         ← Daily rolling log files
├── data/uploads/                 ← Files received via API
├── resumes/                      ← CLI input resumes (git-ignored)
├── jd/                           ← CLI input JDs (git-ignored)
└── template/                     ← DOCX template (read-only)
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: psycopg` | `pip install psycopg[binary]` |
| `connection refused` to PostgreSQL | `docker-compose up -d` (start the container) |
| `pgvector extension not found` | Use `pgvector/pgvector:pg16` image (already in docker-compose) |
| `.doc` file fails to ingest | Install `antiword` (Ubuntu: `sudo apt install antiword`) or `libreoffice` |
| JD returns 0 skills | LLM returned dicts instead of strings — handled by `_normalize_string_list()` |
| Hallucination false positives | Compound skills are split on delimiters; >50% sub-terms grounded = pass |
| SSL certificate errors | Handled automatically (patches httpx SSL for Zscaler proxy) |
| Ollama not running | Script auto-starts it; or run `ollama serve` manually |
| Model not found | Script auto-pulls; or run `ollama pull llama3` |
| API 401/403 | Check `X-API-Key` header matches `AI_MATCHER_API_KEYS` env var |
| Slow ingest (many resumes) | Use `--concurrency 5` for cloud models, keep at 1 for Ollama |

---

## License

Internal use only.
