# AI Resume Matcher — Project Steering

## Overview

Python-based AI-powered resume-to-JD matching engine with a 6-stage pipeline. Supports persistent database storage (SQLite + ChromaDB) with multi-tenant client isolation (NDA enforcement), multi-field embeddings, hybrid search (BM25 + vector), hallucination detection, and scoring evaluation.

## Architecture

Two-phase architecture: **Ingest** (one-time LLM extraction) and **Match** (fast scoring from DB).

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
- `matching_engine/llm_client.py` — LiteLLM wrapper with failover + token estimation
- `matching_engine/database.py` — SQLite profile storage (client-scoped)
- `matching_engine/vector_store.py` — ChromaDB multi-field embeddings + hybrid search
- `matching_engine/scanner.py` — File scanner with TF-IDF keyword extraction + hallucination checks
- `matching_engine/hallucination_check.py` — Grounding verification (LLM output vs source text)
- `matching_engine/evaluation.py` — Scoring validation + bias detection
- `matching_engine/file_loader.py` — Text extraction (PDF/DOCX/TXT, OCR, text boxes)

## Technology Stack

- **Language:** Python 3.10+
- **Data models:** Pydantic v2
- **LLM interface:** LiteLLM (supports Ollama, OpenAI, Anthropic, AWS Bedrock)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (3 collections: skills, experience, summary)
- **Search:** Hybrid — BM25 lexical + vector semantic with Reciprocal Rank Fusion
- **Database:** SQLite (profiles with client_id/job_id)
- **Graph:** NetworkX (knowledge graph)
- **File parsing:** PyPDF2, pdfplumber, python-docx
- **Config:** YAML (PyYAML)
- **Async:** asyncio for concurrent processing
- **Logging:** TimedRotatingFileHandler (daily rollover, 30-day retention)

## Multi-Tenant Isolation (NDA Enforcement)

### Rules

1. Every profile is tagged with `client_id` and `job_id` at ingest time
2. `--client-id` and `--job-id` CLI flags are REQUIRED for `--ingest` and `--match`
3. DB queries ALWAYS filter by `client_id`
4. ChromaDB queries use `where={"client_id": ...}` filter
5. SQLite uses `UNIQUE(client_id, file_hash)` constraint
6. Within same client, resumes are shared across job_ids freely

## Embedding Strategy (Multi-Field)

Three separate ChromaDB collections per resume:
- `resume_skills` — Technologies, tools, certifications (weight: 0.45)
- `resume_experience` — Role titles, companies, domains (weight: 0.35)
- `resume_summary` — Career summary, achievements (weight: 0.20)

Before embedding, TF-IDF keyword extraction removes generic filler words and keeps domain-specific terms.

## Hybrid Search

At query time:
1. JD text is embedded and queried against all 3 collections (vector search)
2. BM25 lexical scoring runs against stored documents (exact keyword matching)
3. Results are fused: 65% vector RRF + 35% BM25
4. Top-N candidates returned for full scoring

## Hallucination Detection

After LLM extraction, `check_hallucination()` verifies:
- Skills: split compound strings, check sub-terms individually (>50% threshold)
- Companies: lenient matching with suffix stripping (Inc, Ltd, etc.)
- Certifications: fuzzy match with aliases (k8s↔kubernetes, AWS↔Amazon Web Services)
- Experience years: cross-check claimed years vs work history date span
- Name: verify name parts appear in source text

Reports confidence score (0-1). Flags unreliable extractions but does not block storage.

## Evaluation Framework

After matching, `generate_evaluation_report()` checks:
- **Ranking stability:** Perturb weights ±10%, check if top-3 changes
- **Score distribution:** Detect all-same, too-narrow, ceiling/floor effects
- **Calibration bands:** Strong/Good/Partial/Weak fit distribution
- **Bias detection:** Experience over-weighting, skill-count correlation
- **Keyword stuffing:** High must_have match but low skills_depth = suspect

## Logging

- Two log files: `logs/ingest.log` and `logs/match.log`
- Daily rollover at midnight, 30-day retention
- All console output (logger + print) captured in log files
- Format: `2026-07-09 14:32:07 | INFO | module_name | message`

## Coding Standards

### Style & Patterns

- Use **type hints** on all function signatures
- Use **Pydantic BaseModel** for all structured data (not raw dicts)
- Use **async/await** for I/O-bound operations (LLM calls, file I/O)
- All modules have docstrings explaining purpose and call relationships
- Use `logging` module for internal state; `print()` for user-facing CLI output
- Configuration priority: CLI flags > config.yaml > built-in defaults

### Error Handling

- LLM calls: retry 3x with exponential backoff, then failover to backup model
- File extraction: fallback from PDF to pdfplumber to OCR; regex baseline for contact info
- JSON parsing from LLM: strip markdown fences, normalize dicts-to-strings, attempt repair
- Never crash the pipeline on a single resume failure; log and continue

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
├── requirements.txt          ← Dependencies
├── resumes/                  ← Input resumes — git-ignored
├── jd/                       ← Input JD files — git-ignored
├── template/                 ← DOCX template (read-only)
├── rendered/                 ← Generated output docs — git-ignored
├── logs/
│   ├── ingest.log            ← Ingest logs (daily rollover)
│   └── match.log             ← Match logs (daily rollover)
├── data/
│   ├── profiles.db           ← SQLite (client-scoped profiles)
│   ├── chroma/               ← ChromaDB (multi-field embeddings)
│   └── scanned_files/        ← Copies of processed resumes
└── matching_engine/
    ├── models.py             ← Pydantic data models
    ├── database.py           ← SQLite with client isolation
    ├── vector_store.py       ← Multi-field ChromaDB + hybrid search
    ├── scanner.py            ← Ingest with TF-IDF + hallucination check
    ├── hallucination_check.py← Grounding verification
    ├── evaluation.py         ← Scoring validation + bias detection
    ├── jd_understanding.py   ← Stage 1
    ├── resume_understanding.py← Stage 2
    ├── semantic_matching.py  ← Stage 3
    ├── scoring.py            ← Stage 4
    ├── explainability.py     ← Stage 5
    ├── template_renderer.py  ← Stage 6
    ├── pipeline.py           ← Pipeline orchestrator
    ├── llm_client.py         ← LiteLLM wrapper
    ├── file_loader.py        ← PDF/DOCX/TXT extraction
    └── utils.py              ← Shared utilities
```

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest resumes for a client
python run.py --ingest --client-id ACME_CORP --job-id JOB-001

# Match against a JD (only that client's resumes visible)
python run.py --match --client-id ACME_CORP --job-id JOB-001

# Combined ingest + match
python run.py --client-id ACME_CORP --job-id JOB-001

# Check DB status (per-client breakdown)
python run.py --db-status

# Bypass DB entirely (stateless mode)
python run.py --scan-mode folder_only
```

## Important Constraints

- **PII sensitivity:** Never commit resumes, JDs, or `data/` to git
- **API keys:** Stored in `.env` (git-ignored). Reference by key name, never expose values
- **Scoring weights** must sum to 1.0
- **Ollama auto-management:** Script auto-starts Ollama and auto-pulls models
- **SSL/proxy handling:** Patches SSL verification for corporate environments (Zscaler)
- **Multi-tenant isolation:** `--client-id` mandatory for all DB modes. Never bypass.
- **Logging:** All output goes to `logs/` with daily rollover

## LLM Prompt Patterns

When modifying LLM prompts:
- Always request JSON output with a strict schema
- Include "respond ONLY with valid JSON" instruction
- Strip markdown code fences from responses before parsing
- Use `_normalize_string_list()` to handle LLM returning dicts instead of strings
- Provide fallback/default values if JSON parsing fails
- Keep temperature low (0.1) for structured extraction
