# AI Resume Matcher — Project Steering

## Overview

This is a Python-based AI-powered resume-to-JD matching engine with a 6-stage pipeline. It supports persistent database storage (SQLite + ChromaDB) so resumes are extracted once and matched instantly against new job descriptions.

**Multi-tenant isolation (NDA enforcement):** Every resume and JD is scoped to a `client_id`. Resumes belonging to one client can NEVER be accessed by another client. Within the same client, resumes are freely shared across `job_id`s.

## Architecture

Two-phase architecture: **Ingest** (one-time, expensive LLM extraction) and **Match** (repeatable, fast scoring from DB).

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

- `run.py` — CLI entry point, orchestrates the full flow
- `matching_engine/pipeline.py` — Pipeline orchestrator (async, concurrent stages)
- `matching_engine/models.py` — Pydantic data models (all structured data)
- `matching_engine/config.py` — AppConfig (Pydantic-based, YAML + CLI merge)
- `matching_engine/llm_client.py` — LiteLLM wrapper with failover + token estimation
- `matching_engine/database.py` — SQLite profile storage & retrieval (client-scoped)
- `matching_engine/vector_store.py` — ChromaDB embedding storage & similarity search (client-scoped)
- `matching_engine/scanner.py` — File scanner with hash-based deduplication (client-scoped)
- `matching_engine/file_loader.py` — Text extraction (PDF/DOCX/TXT, OCR, text boxes)

## Technology Stack

- **Language:** Python 3.10+
- **Data models:** Pydantic v2
- **LLM interface:** LiteLLM (supports Ollama, OpenAI, Anthropic, AWS Bedrock)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2` default)
- **Vector store:** ChromaDB (with client_id metadata filtering)
- **Database:** SQLite (profiles with client_id/job_id columns)
- **Graph:** NetworkX (knowledge graph)
- **File parsing:** PyPDF2, pdfplumber, python-docx
- **Config:** YAML (PyYAML)
- **Async:** asyncio for concurrent processing

## Multi-Tenant Isolation (NDA Enforcement)

### Rules

1. Every resume profile is tagged with `client_id` and `job_id` at ingest time
2. `--client-id` and `--job-id` CLI flags are REQUIRED for `--ingest` and `--match` modes
3. DB queries ALWAYS filter by `client_id` — resumes from one client are NEVER visible to another
4. Within the same `client_id`, resumes are freely shared across `job_id`s
5. ChromaDB queries use `where={"client_id": ...}` filter for vector similarity search
6. SQLite uses `UNIQUE(client_id, file_hash)` constraint — same file can exist under different clients
7. Violating client isolation is treated as an NDA breach — never bypass these checks

### Data Flow

```
--client-id ACME --job-id JOB-001
        │
        ├─→ scanner.py: filters ingested hashes by client_id
        ├─→ database.py: stores profile with client_id + job_id columns
        ├─→ vector_store.py: stores embedding with client_id in metadata
        │
        └─→ At match time:
            ├─→ database.py: SELECT ... WHERE client_id = ?
            └─→ vector_store.py: query(..., where={"client_id": "ACME"})
```

## Coding Standards

### Style & Patterns

- Use **type hints** on all function signatures
- Use **Pydantic BaseModel** for all structured data (not raw dicts)
- Use **async/await** for I/O-bound operations (LLM calls, file I/O)
- All modules have a docstring at the top explaining purpose, usage, and call relationships
- Functions have docstrings with `Called by:` and `Calls:` annotations where relevant
- Use `logging` module (not print) for internal state; `print()` only for user-facing CLI output
- Configuration priority: CLI flags > config.yaml > built-in defaults

### Error Handling

- LLM calls: retry 3x with exponential backoff, then failover to backup model
- File extraction: fallback from PDF to pdfplumber to OCR; regex baseline for contact info
- JSON parsing from LLM: strip markdown fences, attempt repair, log warnings on fallback
- Never crash the pipeline on a single resume failure; log and continue

### Naming Conventions

- Modules: `snake_case.py`
- Classes: `PascalCase` (Pydantic models, pipeline stages)
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: prefix with `_`

### Data Flow

All inter-stage communication uses models from `matching_engine/models.py`:
- `JobDescription` — Stage 1 output (includes `client_id`, `job_id`)
- `ResumeProfile` — Stage 2 output (includes `client_id`, `job_id`)
- `SemanticMatchResult` — Stage 3 output
- `ScoringBreakdown` — Stage 4 output
- `ExplainabilityReport` — Stage 5 output
- `MatchResult` — Final combined result per candidate

## File Organization

```
AI-Resume-Matcher/
├── run.py                    ← CLI entry point
├── config.yaml               ← Runtime configuration
├── requirements.txt          ← Pinned dependencies
├── resumes/                  ← Input resumes (PDF/DOCX/TXT) — git-ignored
├── jd/                       ← Input JD files — git-ignored
├── template/                 ← DOCX template (read-only)
├── rendered/                 ← Generated output docs — git-ignored
├── data/
│   ├── profiles.db           ← SQLite (client-scoped profiles)
│   ├── chroma/               ← ChromaDB (client-scoped embeddings)
│   └── scanned_files/        ← Copies of processed resumes
└── matching_engine/          ← Core library (all pipeline stages)
```

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest resumes for a client (REQUIRED: --client-id and --job-id)
python run.py --ingest --client-id ACME_CORP --job-id JOB-001

# Match against a JD (only ACME_CORP resumes visible)
python run.py --match --client-id ACME_CORP --job-id JOB-001

# Combined ingest + match
python run.py --client-id ACME_CORP --job-id JOB-001

# Check DB status (shows per-client breakdown)
python run.py --db-status

# Bypass DB entirely (original stateless mode, no client-id needed)
python run.py --scan-mode folder_only
```

## Important Constraints

- **PII sensitivity:** Resume files contain personal data. Never commit resumes, JDs, or the `data/` folder to git.
- **API keys:** Stored in `.env` (git-ignored). Reference by key name, never expose values.
- **Scoring weights** must always sum to 1.0 — the config validator enforces this.
- **Ollama auto-management:** The script auto-starts Ollama and auto-pulls models. Don't assume Ollama is running.
- **SSL/proxy handling:** The codebase patches SSL verification for corporate environments (Zscaler). Don't remove those workarounds.
- **No test framework** is currently set up. If adding tests, use `pytest` with `pytest-asyncio`.
- **Multi-tenant isolation (NDA):**
  - `--client-id` and `--job-id` are mandatory for all DB modes
  - Resumes from one client are NEVER returned for another client
  - Violation = NDA breach — never bypass client_id checks in DB or vector store queries

## Databases

### SQLite (`data/profiles.db`)

Stores structured resume profiles. Key columns: `client_id`, `job_id`, `file_hash`, `source_file`, profile fields as JSON.

```bash
sqlite3 data/profiles.db
.schema resume_profiles
SELECT client_id, job_id, source_file, first_name, last_name FROM resume_profiles;
```

### ChromaDB (`data/chroma/`)

File-based vector store for resume embeddings. Each embedding has `client_id` and `job_id` in metadata. Queried via Python API with `where={"client_id": ...}` filter.

## LLM Prompt Patterns

When modifying LLM prompts in `jd_understanding.py` or `resume_understanding.py`:
- Always request JSON output with a strict schema
- Include "respond ONLY with valid JSON" instruction
- Strip markdown code fences from responses before parsing
- Provide fallback/default values if JSON parsing fails
- Keep temperature low (0.1) for structured extraction
