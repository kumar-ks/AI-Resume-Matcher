# AI Resume Matcher — Project Steering

## Overview

This is a Python-based AI-powered resume-to-JD matching engine with a 6-stage pipeline. It supports persistent database storage (SQLite + ChromaDB) so resumes are extracted once and matched instantly against new job descriptions.

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
- `matching_engine/database.py` — SQLite profile storage & retrieval
- `matching_engine/vector_store.py` — ChromaDB embedding storage & similarity search
- `matching_engine/scanner.py` — File scanner with hash-based deduplication
- `matching_engine/file_loader.py` — Text extraction (PDF/DOCX/TXT, OCR, text boxes)

## Technology Stack

- **Language:** Python 3.10+
- **Data models:** Pydantic v2
- **LLM interface:** LiteLLM (supports Ollama, OpenAI, Anthropic, AWS Bedrock)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2` default)
- **Vector store:** ChromaDB
- **Database:** SQLite (profiles)
- **Graph:** NetworkX (knowledge graph)
- **File parsing:** PyPDF2, pdfplumber, python-docx
- **Config:** YAML (PyYAML)
- **Async:** asyncio for concurrent processing

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
- `JobDescription` — Stage 1 output
- `ResumeProfile` — Stage 2 output
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
├── data/                     ← SQLite + ChromaDB + scanned copies — git-ignored
└── matching_engine/          ← Core library (all pipeline stages)
```

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# First-time ingest (requires Ollama or API key)
python run.py --ingest

# Match against a JD (fast, from DB)
python run.py --match

# Full pipeline (ingest new + match)
python run.py

# Check DB status
python run.py --db-status
```

## Important Constraints

- **PII sensitivity:** Resume files contain personal data. Never commit resumes, JDs, or the `data/` folder to git.
- **API keys:** Stored in `.env` (git-ignored). Reference by key name, never expose values.
- **Scoring weights** must always sum to 1.0 — the config validator enforces this.
- **Ollama auto-management:** The script auto-starts Ollama and auto-pulls models. Don't assume Ollama is running.
- **SSL/proxy handling:** The codebase patches SSL verification for corporate environments (Zscaler). Don't remove those workarounds.
- **No test framework** is currently set up. If adding tests, use `pytest` with `pytest-asyncio`.

## LLM Prompt Patterns

When modifying LLM prompts in `jd_understanding.py` or `resume_understanding.py`:
- Always request JSON output with a strict schema
- Include "respond ONLY with valid JSON" instruction
- Strip markdown code fences from responses before parsing
- Provide fallback/default values if JSON parsing fails
- Keep temperature low (0.1) for structured extraction
