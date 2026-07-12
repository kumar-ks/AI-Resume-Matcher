# AI Resume Matcher

AI-powered Resume to Job Description matching engine with a 6-stage pipeline. Now supports **persistent database storage** — resumes are extracted once and stored, then matched instantly against new JDs.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Two Operating Modes](#two-operating-modes)
- [CLI Options](#cli-options)
- [Configuration (`config.yaml`)](#configuration-configyaml)
- [Scan Modes](#scan-modes)
- [Architecture](#architecture)
- [Speed Comparison](#speed-comparison)
- [Model Failover](#model-failover)
- [Scoring Weights — Design Rationale](#scoring-weights--design-rationale)
- [Template-Based Resume Generation (`--generate-doc`)](#template-based-resume-generation---generate-doc)
- [API Keys (`.env` file)](#api-keys-env-file)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Example Usage](#example-usage)
- [Example Output](#example-output)

---

## Quick Start

### 1. Install dependencies

```bash
cd AI-Resume-Matcher
pip install -r requirements.txt
```

### 2. Install Ollama (for local LLM)

```bash
brew install ollama
ollama pull llama3
```

> The script auto-starts Ollama if not running. It also auto-pulls the model if not downloaded yet.

### 3. Place your files

- Drop resumes into the `resumes/` folder (PDF, DOCX, TXT)
- Place the job description into the `jd/` folder (PDF, DOCX, TXT)

### 4. Configure

Edit `config.yaml` to set your preferred model, paths, and scan mode:

```yaml
model: "anthropic/claude-3-sonnet-20240229"
scan_mode: "db_first"
concurrency: 3
```

### 5. Run

```bash
# First time: process resumes into the database (requires --client-id and --job-id)
python run.py --ingest --client-id ACME_CORP --job-id JOB-001

# Match against a JD (fast, reads from DB, client-scoped)
python run.py --match --client-id ACME_CORP --job-id JOB-001

# Or combined: ingests new resumes + matches from DB
python run.py --client-id ACME_CORP --job-id JOB-001
```

---

## Multi-Tenant Client Isolation (NDA)

The system enforces strict client-level data isolation. This is an NDA requirement — resumes belonging to one client can **never** be accessed, matched, or returned when processing a JD for a different client.

### How It Works

| Rule | Enforcement |
|------|-------------|
| Resumes for Client A are invisible to Client B | DB queries filter by `client_id` (SQL WHERE clause + ChromaDB `where` filter) |
| Within the same client, resumes are shared across jobs | `job_id` is stored for tracking but not used as an isolation boundary |
| `--client-id` and `--job-id` are mandatory | CLI validation exits with error if missing |
| Same resume file can exist under different clients | `UNIQUE(client_id, file_hash)` constraint in SQLite |

### Usage

```bash
# Ingest resumes for Client A
python run.py --ingest --client-id CLIENT_A --job-id JOB-101

# Match for Client A (only Client A resumes visible)
python run.py --match --client-id CLIENT_A --job-id JOB-102

# Ingest resumes for Client B (completely separate pool)
python run.py --ingest --client-id CLIENT_B --job-id JOB-201

# Match for Client B (Client A resumes are NEVER visible here)
python run.py --match --client-id CLIENT_B --job-id JOB-201

# Check breakdown per client
python run.py --db-status
```

### Where Isolation Is Enforced

- **SQLite** (`database.py`): All read methods require `client_id` parameter. Raises `ValueError` if empty.
- **ChromaDB** (`vector_store.py`): `query_similar()` uses `where={"client_id": ...}` filter. Raises `ValueError` if empty.
- **Scanner** (`scanner.py`): Deduplication is scoped per client. Same file can be ingested under different clients.
- **CLI** (`run.py`): `_validate_tenant_flags()` blocks execution if flags are missing.

Edit `config.yaml` to set your preferred model, paths, and scan mode:

```yaml
model: "anthropic/claude-3-sonnet-20240229"
scan_mode: "db_first"
concurrency: 3
```

### 5. Run

```bash
# First time: process resumes into the database
python run.py --ingest

# Match against a JD (fast, reads from DB)
python run.py --match

# Or combined: ingests new resumes + matches from DB (default)
python run.py
```

---

## Two Operating Modes

### Ingest Mode (`--ingest`)

Processes resumes and stores them persistently. Run once (or whenever new resumes are added).

| Step | What happens |
|------|--------------|
| 1 | Scans `resumes/` folder for files |
| 2 | Checks file hash against DB (skips already-processed files) |
| 3 | Extracts profile via LLM (Stage 2) |
| 4 | Stores structured profile in SQLite |
| 5 | Stores embeddings in ChromaDB |
| 6 | Copies original file to `data/scanned_files/` |

Ingest is async and non-blocking — processes multiple resumes in parallel (controlled by `--concurrency`).

### Match Mode (`--match`)

Matches stored profiles against a JD. No re-extraction needed — this is where the speed comes from.

| Step | What happens |
|------|--------------|
| 1 | Loads JD, extracts requirements via LLM (one call) |
| 2 | Queries ChromaDB for top similar profiles (semantic pre-filter) |
| 3 | Loads full profiles from SQLite |
| 4 | Runs scoring (Stage 3-4, no LLM needed) |
| 5 | Explains top N candidates (Stage 5, optional LLM) |

**~20 seconds for 100 profiles** when matching from DB.

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `./config.yaml` | Path to config file |
| `--resumes` | `./resumes` | Folder with resume files |
| `--jd` | (auto from `./jd/`) | Specific JD file path |
| `--jd-dir` | `./jd` | Folder with JD files |
| `--model` | from config | LLM model (e.g., `ollama/llama3`, `gpt-4`) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Embedding model for semantic matching |
| `--top` | all | Show only top N candidates |
| `--output` | none | Save JSON results to file |
| `--debug` | off | Verbose logging |
| `--concurrency` | 3 | Parallel processing count |
| `--explain-top` | all | Only explain top N candidates |
| `--generate-doc` | none | Generate DOCX for top N candidates |
| `--ingest` | off | Ingest resumes into DB |
| `--match` | off | Match from DB against JD |
| `--scan-mode` | `db_first` | `db_first` / `folder_only` / `db_only` |
| `--db-status` | off | Show DB stats and exit |

**Priority order:** `CLI flags` > `config.yaml` > `built-in defaults`

---

## Configuration (`config.yaml`)

```yaml
# ─────────────────────────────────────────────────────────────────────────────
# AI Resume Matcher — Configuration File
# ─────────────────────────────────────────────────────────────────────────────
# All settings can be overridden via CLI flags (CLI takes priority over config).
# Priority: CLI flags > config.yaml > built-in defaults
# ─────────────────────────────────────────────────────────────────────────────

# ── LLM Model Configuration ──────────────────────────────────────────────────
# Primary model: Used first for all LLM calls.
# Failover model: Automatically used if primary fails (API error, rate limit, timeout).
#
# Supported providers via LiteLLM:
#   Local:    ollama/llama3, ollama/mistral, ollama/qwen2, ollama/llama3:70b
#   OpenAI:   gpt-4, gpt-4o, gpt-3.5-turbo
#   Anthropic: anthropic/claude-3-sonnet-20240229, anthropic/claude-3-opus-20240229
#   AWS:      bedrock/anthropic.claude-3-sonnet
model: "anthropic/claude-3-sonnet-20240229"
failover_model: "ollama/llama3"

# ── Embedding Model ──────────────────────────────────────────────────────────
# Sentence-transformers model for semantic matching (Stage 3).
#   Fast:     all-MiniLM-L6-v2 (80MB, good balance)
#   Accurate: all-mpnet-base-v2 (420MB, better quality)
embedding_model: "all-MiniLM-L6-v2"

# ── File Paths ────────────────────────────────────────────────────────────────
resumes_dir: "./resumes"
jd_dir: "./jd"

# ── Performance ──────────────────────────────────────────────────────────────
# concurrency: Resumes processed in parallel
#   Ollama (local): 2-3 (limited by GPU/CPU)
#   Cloud APIs:     5-10 (rate limit dependent)
concurrency: 3

# explain_top: Only generate AI explanations for top N candidates
#   null = explain all | 10 = recommended for 50+ resumes
explain_top: null

# ── LLM Parameters ───────────────────────────────────────────────────────────
temperature: 0.1
max_tokens: 4096

# ── Scoring Weights ──────────────────────────────────────────────────────────
# Must sum to 1.0. Adjust based on hiring criteria.
# See Scoring Weights section below for detailed reasoning.
scoring_weights:
  must_have_match: 0.35       # "Can they do the job?" (hard filter)
  experience_match: 0.25      # "Are they at the right level?"
  skills_depth: 0.20          # "How deep is their expertise?"
  project_relevance: 0.12     # "Have they applied it practically?"
  recency_factor: 0.08        # "Is their experience current?"

# ── Output ────────────────────────────────────────────────────────────────────
output_file: null       # Set to "results.json" to always export
top_n: null             # Show only top N (null = show all)
debug: false            # Enable verbose logging

# ── Database & Scanner ────────────────────────────────────────────────────────
# Storage paths for the persistent resume database
db_path: "./data/profiles.db"              # SQLite profile storage
vector_store_path: "./data/chroma"         # ChromaDB embedding storage
scanned_files_path: "./data/scanned_files" # Copies of processed resumes

# scan_mode controls how matching finds candidates:
#   "db_first"    — Ingests new files + matches from DB (DEFAULT)
#   "folder_only" — Bypass DB, scan folder directly (original behavior)
#   "db_only"     — Only use DB, never scan folder (fastest)
scan_mode: "db_first"
```

---

## Scan Modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `db_first` (default) | Ingests new files into DB, then matches from DB | Normal operation |
| `folder_only` | Bypasses DB entirely, processes fresh every time (original behavior) | Testing, debugging, bypass |
| `db_only` | Only uses what's already in the DB (fastest, no folder scan) | Repeated queries against same pool |

Set via config or CLI:

```bash
python run.py --scan-mode db_only --match
```

---

## Architecture

### Two-Phase Architecture

The system separates resume processing (expensive, one-time) from matching (cheap, repeatable):

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PHASE 1: INGEST (one-time)                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  resumes/           File Loader         LLM Extraction       Storage       ║
║  ┌──────────┐      ┌───────────┐      ┌──────────────┐    ┌──────────┐   ║
║  │ PDF/DOCX │─────▶│ extract   │─────▶│ Stage 2:     │───▶│ SQLite   │   ║
║  │ TXT/IMG  │      │ text      │      │ Profile JSON │    │ ChromaDB │   ║
║  └──────────┘      └───────────┘      └──────────────┘    └──────────┘   ║
║                                                                            ║
║  • Checks file hash (skips duplicates)                                     ║
║  • Parallel processing (--concurrency)                                     ║
║  • Copies originals to data/scanned_files/                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                     PHASE 2: MATCH (repeatable, fast)                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  jd/              LLM Extraction     ChromaDB Query      Scoring           ║
║  ┌──────────┐    ┌──────────────┐   ┌─────────────┐   ┌──────────────┐   ║
║  │ JD File  │───▶│ Stage 1:     │──▶│ Semantic    │──▶│ Stage 3-4:   │   ║
║  │          │    │ Requirements │   │ Pre-filter  │   │ Score + Rank │   ║
║  └──────────┘    └──────────────┘   └──────┬──────┘   └──────┬───────┘   ║
║                                             │                  │           ║
║                                      ┌──────▼──────┐   ┌──────▼───────┐   ║
║                                      │ SQLite:     │   │ Stage 5:     │   ║
║                                      │ Full Profile│   │ Explanations │   ║
║                                      └─────────────┘   └──────────────┘   ║
║                                                                            ║
║  • Single LLM call for JD                                                  ║
║  • No LLM needed for scoring (Stage 3-4)                                   ║
║  • Optional LLM for explanations (Stage 5, top N only)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Pipeline Stages

| Stage | Name | LLM Required | Description |
|-------|------|:------------:|-------------|
| 1 | JD Understanding | Yes (1 call) | Extracts structured requirements from JD |
| 2 | Resume Understanding | Yes (per resume) | Extracts structured profile (ingest only) |
| 3 | Semantic Matching | No | Embedding similarity across 6 dimensions |
| 4 | Scoring | No | Weighted formula → qualification percentage |
| 5 | Explainability | Optional | LLM reasoning for top N, rule-based for rest |
| 6 | Template Rendering | No | DOCX generation (if `--generate-doc`) |

---

## Speed Comparison

| Scenario | Folder-only mode | DB mode |
|----------|-----------------|---------|
| 5 resumes, first run | ~3 min | ~3 min (ingest) |
| 5 resumes, new JD | ~3 min | ~20 sec |
| 100 resumes, first run | ~40 min | ~40 min (ingest) |
| 100 resumes, new JD | ~40 min | ~30 sec |

The entire point of DB mode: **pay the extraction cost once**, then match against unlimited JDs in seconds.

---

## Model Failover

The framework supports automatic failover between models:

```
Primary Model (config)  ───FAIL───▶  Failover Model (config)
       │                                      │
       │ SUCCESS                              │ SUCCESS
       ▼                                      ▼
  Pipeline proceeds                    Pipeline proceeds
```

### Configuration

```yaml
model: "anthropic/claude-3-sonnet-20240229"   # Primary (tried first)
failover_model: "ollama/llama3"                # Fallback (used if primary fails)
```

### Pre-flight validation

Before the pipeline starts:
1. Checks if the API key is set (for cloud models)
2. Pings the primary model to verify availability
3. If primary is down → pings failover model
4. Reports which model will be used

### Failover triggers

- Missing API key (`AuthenticationError`)
- Network timeout or connection error
- Rate limiting (`429 Too Many Requests`)
- Model not found or unavailable

Once failed over, all subsequent calls use the failover model (no ping-pong).

---

## Scoring Weights — Design Rationale

### The 5 Dimensions

These map to the 5 signals a recruiter evaluates when screening resumes:

| Dimension | Weight | Recruiter's Question |
|-----------|--------|---------------------|
| `must_have_match` | 0.35 | "Does this person have the required skills?" |
| `experience_match` | 0.25 | "Do they have enough years at this level?" |
| `skills_depth` | 0.20 | "Do they actually know these technologies deeply?" |
| `project_relevance` | 0.12 | "Have they built relevant things?" |
| `recency_factor` | 0.08 | "Is their experience current?" |

### Why These Specific Weights?

- **35% must-have skills** — The gatekeeper. If a candidate lacks 3 of 5 must-have skills, they're out regardless of experience. Highest weight because it's a hard filter.
- **25% experience** — Seniority matters. A Lead role needs someone who's led teams, not a fresh grad with matching keywords.
- **20% depth** — Separates "I've used Docker once" from "I've built production Kubernetes clusters." Uses semantic embeddings to detect depth beyond exact keyword matching.
- **12% projects** — Practical evidence. Lower weight because not all resumes list projects (senior engineers often describe work in responsibilities instead).
- **8% recency** — A tiebreaker. If two candidates score the same on everything else, the one with more recent experience wins. Low weight because skills don't expire quickly.

### Customizing for Different Role Types

```yaml
# Senior IC role (skills matter most):
scoring_weights:
  must_have_match: 0.40
  experience_match: 0.20
  skills_depth: 0.25
  project_relevance: 0.10
  recency_factor: 0.05

# Leadership role (experience matters most):
scoring_weights:
  must_have_match: 0.25
  experience_match: 0.35
  skills_depth: 0.15
  project_relevance: 0.15
  recency_factor: 0.10

# Junior role (potential over experience):
scoring_weights:
  must_have_match: 0.30
  experience_match: 0.10
  skills_depth: 0.25
  project_relevance: 0.25
  recency_factor: 0.10
```

**The only rule:** weights must sum to 1.0. If they don't, the scorer auto-normalizes and logs a warning.

---

## Template-Based Resume Generation (`--generate-doc`)

After the pipeline ranks candidates, you can auto-generate formatted DOCX documents using your company's resume template.

### How It Works

1. **Template is read-only** — Your template in `template/` is never modified.
2. **Data is filled** — Candidate profile data (name, contact, summary, skills, experience, education, certifications) is populated into a fresh copy.
3. **Output is saved** — `rendered/{rank}_Antern_{original_filename}.docx`

### Usage

```bash
# Generate for top 3 candidates
python run.py --match --generate-doc 3

# Combine with JSON export
python run.py --match --generate-doc 3 --output results.json
```

### What Gets Filled

| Template Section | Data Source |
|-----------------|-------------|
| NAME | `ResumeProfile.full_name` (uppercase) |
| Contact / Email | `ResumeProfile.phone`, `email`, `location` |
| PROFESSIONAL SUMMARY | `ResumeProfile.career_summary` |
| TECHNICAL SKILLS | `ResumeProfile.skills[]` (comma-separated) |
| EXPERIENCE | `ResumeProfile.work_experiences[]` (title, company, dates, responsibilities) |
| EDUCATION | `ResumeProfile.education[]` |
| CERTIFICATIONS | `ResumeProfile.certifications[]` |

### Setup

Place your company's DOCX template in the `template/` folder. The first `.docx` file found is used. The template should have section headers (PROFESSIONAL SUMMARY, TECHNICAL SKILLS, EXPERIENCE, EDUCATION, CERTIFICATIONS) as markers.

---

## API Keys (`.env` file)

### Setup

```bash
cp .env.example .env
```

Then edit `.env` and add your keys:

```bash
# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# OpenAI (GPT-4)
OPENAI_API_KEY=sk-your-key-here

# AWS Bedrock (if using bedrock/ models)
AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_REGION_NAME=us-east-1
```

Keys are auto-loaded by the script via `python-dotenv`. The `.env` file is git-ignored — never commit it.

> If using Ollama (local model), no API keys are needed.

---

## Project Structure

```
AI-Resume-Matcher/
├── run.py                              ← Main entry point
├── config.yaml                         ← Configuration (model, paths, weights, scan_mode)
├── .env / .env.example                 ← API keys (git-ignored / template)
├── requirements.txt                    ← Python dependencies
├── .gitignore                          ← Git ignore rules
├── README.md                           ← This file
├── resumes/                            ← Place candidate resumes here (PDF/DOCX/TXT)
├── jd/                                 ← Place JD file here (PDF/DOCX/TXT)
├── template/                           ← DOCX template (read-only, never modified)
├── rendered/                           ← Generated formatted docs (auto-created)
├── data/
│   ├── profiles.db                     ← SQLite profile storage
│   ├── chroma/                         ← ChromaDB embedding storage
│   └── scanned_files/                  ← Copies of processed resumes
└── matching_engine/
    ├── __init__.py
    ├── models.py                       ← Pydantic data models
    ├── config.py                       ← AppConfig (settings validation)
    ├── utils.py                        ← Shared utilities (JSON parsing, text helpers)
    ├── file_loader.py                  ← Text extraction (PDF/DOCX/TXT, OCR, text boxes)
    ├── llm_client.py                   ← LLM client with failover + token estimation
    ├── database.py                     ← NEW: SQLite profile storage & retrieval
    ├── vector_store.py                 ← NEW: ChromaDB embedding storage & similarity search
    ├── scanner.py                      ← NEW: File scanner with hash-based deduplication
    ├── jd_understanding.py             ← Stage 1: LLM-based JD parsing
    ├── resume_understanding.py         ← Stage 2: Regex + LLM resume parsing
    ├── semantic_matching.py            ← Stage 3: Embedding similarity (6 dimensions)
    ├── scoring.py                      ← Stage 4: Weighted scoring formula
    ├── explainability.py               ← Stage 5: LLM explanation + rule-based fallback
    ├── pipeline.py                     ← Pipeline orchestrator (concurrent stages)
    └── template_renderer.py            ← DOCX template rendering (--generate-doc)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| JD shows 0 characters | Scanned/image-based PDF | Install OCR: `brew install tesseract poppler && pip install pytesseract pdf2image` |
| All scores identical | Ollama not running or JD empty | Script auto-starts Ollama. Check if JD has extractable text. |
| SSL certificate errors | Corporate proxy (Zscaler) | Handled automatically (sets `LITELLM_LOCAL_MODEL_COST_MAP=True`) |
| Embedding model fails | SSL blocks HuggingFace download | Framework auto-patches httpx SSL verification |
| LLM returns bad JSON | Model too small | Use `ollama/llama3` or larger; framework retries 3× with fallbacks |
| DOCX shows minimal text | Content in text boxes/tables | Framework extracts from paragraphs, tables, text boxes, hyperlinks |
| Name/email/phone missing | LLM failed extraction | Regex baseline always extracts contact info as fallback |
| Ollama not installed | Binary not found | `brew install ollama` or https://ollama.com/download |
| Model not found | Not pulled yet | Script auto-pulls on first run, or: `ollama pull llama3` |
| Slow (100+ resumes) | Processing sequentially | Use `--concurrency 5 --explain-top 10` |
| DB match returns 0 | No resumes ingested yet | Run `python run.py --ingest` first |
| Duplicate profiles in DB | Same resume processed twice | Hash-based dedup prevents this; use `--db-status` to check |
| ChromaDB errors | Corrupted vector store | Delete `data/chroma/` and re-ingest: `python run.py --ingest` |
| `--match` is slow | Explain-all enabled for many profiles | Use `--explain-top 10` to limit LLM explanations |

---

## Example Usage

```bash
# ── First time setup ──────────────────────────────────────────────────────────

# Process all resumes into the database (requires --client-id and --job-id)
python run.py --ingest --client-id ACME_CORP --job-id JOB-001

# Check what's in the database (shows per-client breakdown)
python run.py --db-status

# ── Day-to-day usage ─────────────────────────────────────────────────────────

# Match against a new JD (fast, client-scoped, ~20 seconds from DB)
python run.py --match --client-id ACME_CORP --job-id JOB-002

# Match with only top 5 results shown
python run.py --match --client-id ACME_CORP --job-id JOB-002 --top 5

# Full pipeline: ingest new resumes + match from DB (default mode)
python run.py --client-id ACME_CORP --job-id JOB-001

# ── Multi-tenant usage ───────────────────────────────────────────────────────

# Ingest for a different client (completely isolated pool)
python run.py --ingest --client-id CLIENT_B --job-id JOB-201

# Match for Client B (ACME_CORP resumes are NEVER visible)
python run.py --match --client-id CLIENT_B --job-id JOB-201

# ── Advanced usage ───────────────────────────────────────────────────────────

# Bypass DB entirely (original folder-scan behavior, no client-id needed)
python run.py --scan-mode folder_only

# Only use what's in DB, never touch the folder
python run.py --scan-mode db_only --match --client-id ACME_CORP --job-id JOB-001

# Generate formatted DOCX for top 3 + export JSON
python run.py --match --generate-doc 3 --output results.json

# Use a different model with higher concurrency
python run.py --model gpt-4 --concurrency 8 --explain-top 10

# Debug mode (verbose logging)
python run.py --debug --match
```

---

## Example Output

### Terminal (Summary Table)

```
⏱  Pipeline completed in 18.7 seconds (5 profiles from DB)

====================================================================================================================
RESULTS — Candidate Match Grid (sorted by % Qualified)
====================================================================================================================
#   Source File                     First Name   Last Name       Exp(Yrs)  % Match  Key Skills (Top 3)                  Action
--------------------------------------------------------------------------------------------------------------------
1   DevSecOps_MLOps_v3.docx         Kumar        Karpuram        17.0      59.7%    MLOps, Docker, Kubernetes            👍 Consider for interview
2   Kumar_DevSecOps_MLOps_v1.pdf    Kumar        Karpuram        13.0      56.6%    Python, AWS, Terraform               👍 Consider for interview
3   Jyothi Kancharla.docx           Jyothi       Kancharla       15.0      52.2%    Data Science, Python, NLP            👍 Consider for interview
4   Arun Prasad Resume.pdf          Arun         Prasad          17.0      50.9%    Java, Spring, Microservices          ⚠️  May need additional screening
5   Resume_Sr.Engineering_Spec...   Kumar        K               13.0      32.0%    Selenium, Java, DevOps               ⚠️  May need additional screening
====================================================================================================================
```

---

## License

Internal use only.
