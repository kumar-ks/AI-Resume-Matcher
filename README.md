# AI Resume Matcher

AI-powered Resume to Job Description matching engine with a 6-stage pipeline. Uses LLM-based extraction, embedding-based semantic matching, and weighted scoring to rank candidates against a job description.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration (`config.yaml`)](#configuration-configyaml)
- [CLI Options](#cli-options)
- [LLM Providers](#llm-providers)
- [Performance Tuning](#performance-tuning)
- [Model Failover](#model-failover)
- [Scoring Weights — Design Rationale](#scoring-weights--design-rationale)
- [JSON Output (for UI)](#json-output-for-ui)
- [Framework Architecture](#framework-architecture)
- [Execution Flow (Sequence of Calls)](#execution-flow-sequence-of-calls)
- [File-by-File Summary](#file-by-file-summary)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Example Output](#example-output)

---

## Quick Start

### 1. Install dependencies

```bash
cd AI-Resume-Matcher
pip install -r requirements.txt
```

### 2. Install Ollama (for local LLM)

Ollama runs LLMs locally — no API keys, no data leaves your machine.

```bash
# macOS (via Homebrew)
brew install ollama

# Or download from: https://ollama.com/download
```

Then pull the model (one-time download, ~4.7 GB):

```bash
ollama pull llama3
```

> **Note:** The script auto-starts Ollama if it's not running. You don't need to manually run `ollama serve` — the framework handles it automatically. It also auto-pulls the model if not downloaded yet.

### 3. Place your files

```
AI-Resume-Matcher/
├── resumes/                ← Drop all candidate resumes here (PDF/DOCX/TXT)
├── jd/                     ← Place the job description here (PDF/DOCX/TXT)
├── config.yaml             ← Configuration (model, paths, weights, performance)
└── run.py                  ← Run this
```

### 4. Configure (optional)

Edit `config.yaml` to set your preferred model and settings:

```yaml
model: "ollama/llama3"      # Change model here — no CLI flag needed
concurrency: 3              # Parallel resume processing
explain_top: 10             # Only explain top 10 candidates (saves time)
```

### 5. Run

```bash
# Uses config.yaml settings (model, paths, concurrency all from config)
python run.py

# Override any setting via CLI (CLI takes priority over config)
python run.py --model gpt-4
python run.py --concurrency 5 --explain-top 10
python run.py --output results.json

# Debug mode (verbose logging)
python run.py --debug
```

### What happens when you run the script

1. **Config loaded** — Reads `config.yaml` for model, paths, and performance settings.
2. **Ollama auto-start** — If using an `ollama/*` model and the server isn't running, starts it automatically.
3. **Model check** — If the required model isn't downloaded yet, pulls it automatically (first run only).
4. **File loading** — Reads JD and resumes from configured folders.
5. **Pipeline execution** — Runs the 6-stage matching pipeline (with concurrent processing).
6. **Results display** — Shows ranked candidates with full details.
7. **JSON export** — Saves results to JSON for UI consumption (if `--output` specified).

---

## Configuration (`config.yaml`)

All settings live in one file. CLI flags override config values.

```yaml
# ─────────────────────────────────────────────────────────────────────────────
# AI Resume Matcher — Configuration File
# ─────────────────────────────────────────────────────────────────────────────
# Priority: CLI flags > config.yaml > built-in defaults
# ─────────────────────────────────────────────────────────────────────────────

# ── LLM Model ────────────────────────────────────────────────────────────────
# Local:    ollama/llama3, ollama/mistral, ollama/qwen2, ollama/llama3:70b
# OpenAI:   gpt-4, gpt-4o, gpt-3.5-turbo
# Anthropic: anthropic/claude-3-sonnet-20240229
# AWS:      bedrock/anthropic.claude-3-sonnet
model: "ollama/llama3"

# ── Embedding Model ──────────────────────────────────────────────────────────
# Fast:     all-MiniLM-L6-v2 (80MB, good balance)
# Accurate: all-mpnet-base-v2 (420MB, better quality)
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

# ── Scoring Weights (must sum to 1.0) ────────────────────────────────────────
scoring_weights:
  must_have_match: 0.35
  experience_match: 0.25
  skills_depth: 0.20
  project_relevance: 0.12
  recency_factor: 0.08

# ── Output ────────────────────────────────────────────────────────────────────
output_file: null       # Set to "results.json" to always export
top_n: null             # Show only top N (null = show all)
debug: false            # Enable verbose logging
```

**Priority order:** `CLI flags` > `config.yaml` > `built-in defaults`

---

## CLI Options

| Flag | Default | Config key | Description |
|------|---------|------------|-------------|
| `--config` | `./config.yaml` | — | Path to YAML config file |
| `--resumes` | `./resumes` | `resumes_dir` | Folder containing resume files |
| `--jd` | (auto) | — | Path to specific JD file |
| `--jd-dir` | `./jd` | `jd_dir` | Folder containing JD file(s) |
| `--model` | `ollama/llama3` | `model` | LLM model identifier |
| `--embedding-model` | `all-MiniLM-L6-v2` | `embedding_model` | Sentence-transformers model |
| `--top` | all | `top_n` | Show only top N candidates |
| `--output` | none | `output_file` | Save results to JSON file |
| `--concurrency` | 3 | `concurrency` | Parallel resume processing count |
| `--explain-top` | all | `explain_top` | Only AI-explain top N candidates |
| `--debug` | off | `debug` | Enable DEBUG-level logging |

---

## LLM Providers

Uses [LiteLLM](https://github.com/BerriAI/litellm) — supports 100+ providers via a single interface:

| Provider | Config `model` value | Env var needed |
|----------|---------------------|----------------|
| Ollama (local) | `ollama/llama3` | None (auto-starts) |
| OpenAI | `gpt-4` or `gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-3-sonnet-20240229` | `ANTHROPIC_API_KEY` |
| AWS Bedrock | `bedrock/anthropic.claude-3-sonnet` | AWS credentials |

To switch models, edit `config.yaml`:
```yaml
model: "gpt-4"    # Just change this line
```

---

## Performance Tuning

| Resumes | Recommended config | Estimated time |
|---------|-------------------|----------------|
| 5 | `concurrency: 3` | ~2 min (Ollama) |
| 50 | `concurrency: 3, explain_top: 10` | ~12 min (Ollama) |
| 100 | `concurrency: 5, explain_top: 10` | ~20 min (Ollama) |
| 100 | `concurrency: 10, explain_top: 10` (GPT-4) | ~5 min |

**Key optimizations built into the pipeline:**

1. **Concurrent processing** — Multiple resumes processed in parallel (configurable via `concurrency`).
2. **Deferred explainability** — Stage 5 (LLM explanation) runs AFTER sorting. Low-scoring candidates get fast rule-based explanations instead of expensive LLM calls.
3. **`--explain-top N`** — Only generates AI explanations for the top N candidates. Saves ~15s per skipped candidate.
4. **Embedding pre-loading** — The sentence-transformers model is loaded once at startup, not per-resume.

```bash
# Fast batch processing (100 resumes)
python run.py --concurrency 5 --explain-top 10 --output results.json

# Maximum speed with cloud API
python run.py --model gpt-4 --concurrency 10 --explain-top 20 --output results.json
```

---

## Model Failover

The framework supports automatic failover between models. If the primary model (e.g., Claude) is unavailable, it switches to the failover model (e.g., Ollama) transparently.

### How it works

```
┌─────────────────┐     FAIL      ┌─────────────────┐
│  Primary Model  │──────────────▶│  Failover Model │
│  (Claude/GPT-4) │               │  (Ollama local) │
└────────┬────────┘               └────────┬────────┘
         │ SUCCESS                          │ SUCCESS
         ▼                                  ▼
   ┌───────────┐                     ┌───────────┐
   │  Pipeline │                     │  Pipeline │
   │  proceeds │                     │  proceeds │
   └───────────┘                     └───────────┘
```

### Configuration

In `config.yaml`:
```yaml
model: "anthropic/claude-3-sonnet-20240229"   # Primary (tried first)
failover_model: "ollama/llama3"                # Fallback (used if primary fails)
```

### Pre-flight validation

Before the pipeline starts, the framework:
1. Checks if the API key is set (for cloud models)
2. Pings the primary model to verify availability
3. If primary is down → pings failover model
4. Reports which model will be used

```
Validating LLM access...
  ⚠️  Primary 'anthropic/claude-3-sonnet' unavailable. Using failover 'ollama/llama3'.
```

### Failover triggers

The failover activates on:
- Missing API key (`AuthenticationError`)
- Network timeout or connection error
- Rate limiting (`429 Too Many Requests`)
- Model not found or unavailable

Once failed over, all subsequent calls use the failover model (no ping-pong).

---

## Scoring Weights — Design Rationale

### Why these 5 dimensions?

These map to the **5 signals a recruiter evaluates** when screening resumes:

| Dimension | Recruiter's question | Why it matters |
|-----------|---------------------|----------------|
| `must_have_match` (35%) | "Does this person have the required skills?" | Hard filter — lacking core skills is a disqualifier |
| `experience_match` (25%) | "Do they have enough years at this level?" | A 2-year dev won't fit a Lead role |
| `skills_depth` (20%) | "Do they actually know these technologies deeply?" | Listing "Python" ≠ building production ML pipelines |
| `project_relevance` (12%) | "Have they built relevant things?" | Proves practical application, not just keywords |
| `recency_factor` (8%) | "Is their experience current?" | Java in 2010 is less relevant than Java in 2024 |

### Why these specific weights?

```
must_have_match:  0.35  ← HIGHEST: Hard filter — "Can they do the job?"
experience_match: 0.25  ← HIGH: Level alignment — "Are they senior enough?"
skills_depth:     0.20  ← MEDIUM: Beyond keywords — "How deep is their expertise?"
project_relevance:0.12  ← LOWER: Evidence — "Have they applied it?"
recency_factor:   0.08  ← LOWEST: Tiebreaker — "Is it recent?"
```

**The reasoning:**
- **35% must-have skills** — The gatekeeper. If a candidate lacks 3 of 5 must-have skills, they're out regardless of experience. Gets the highest weight.
- **25% experience** — Seniority level matters heavily. A Lead role needs someone who's led teams, not a fresh graduate with matching keywords.
- **20% depth** — Separates "I've used Docker once" from "I've built production Kubernetes clusters." Uses semantic embeddings to detect this beyond exact keyword matching.
- **12% projects** — Practical evidence. Lower weight because not all resumes list projects (especially senior engineers who describe work in responsibilities instead).
- **8% recency** — A tiebreaker. If two candidates score the same on everything else, the one with more recent relevant experience wins. Low weight because skills don't expire quickly.

### Customizing for different roles

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

The only rule: **weights must sum to 1.0**. If they don't, the scorer auto-normalizes and logs a warning.

---

## JSON Output (for UI)

The `--output results.json` flag produces a structured JSON file designed for UI consumption. Each candidate entry maps back to its source resume file.

```bash
python run.py --output results.json
```

**JSON structure:**

```json
{
  "metadata": {
    "jd_file": "Lead_MLOps_Engineer.pdf",
    "jd_title": "Lead MLOps Engineer",
    "total_candidates": 5,
    "model_used": "ollama/llama3",
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "candidates": [
    {
      "rank": 1,
      "source_file": "DevSecOps_MLOps_v3.docx",
      "source_path": "/full/path/to/resumes/DevSecOps_MLOps_v3.docx",

      "first_name": "Kumar",
      "middle_name": "S",
      "last_name": "Karpuram",
      "full_name": "Kumar S Karpuram",
      "contact_number": "+91-96864-88688",
      "email": "shootmail2kumar@gmail.com",
      "location": "Bangalore, India",
      "total_experience_years": 17.0,

      "qualification_percentage": 59.7,
      "action": "Good Fit - Consider for interview",
      "reasoning": "Strong match in MLOps and cloud platform skills...",

      "key_skills_top_5": ["MLOps", "Docker", "Kubernetes", "Python", "AWS"],
      "all_skills": ["MLOps", "Docker", "Kubernetes", "Python", "AWS", "..."],
      "matched_strengths": ["Python expertise", "K8s experience", "..."],
      "missing_skills": ["Data Science depth", "..."],

      "scoring_breakdown": {
        "must_have_match": 0.667,
        "experience_match": 0.800,
        "skills_depth": 0.534,
        "project_relevance": 0.600,
        "recency_factor": 0.800
      },

      "work_experiences": [
        {
          "company": "BT Group",
          "title": "Sr. Engineering Specialist",
          "start_year": 2022,
          "end_year": null,
          "is_current": true,
          "technologies": ["Kubernetes", "AWS", "Docker", "Python"]
        }
      ],

      "education": ["M.C.A - Bangalore University"],
      "certifications": ["AWS Solutions Architect"]
    }
  ]
}
```

**UI mapping:** Use `source_file` and `source_path` to link each result back to the original resume (e.g., for download links, document preview, or file viewer).

---

## Framework Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              run.py (Entry Point)                            │
│  Loads config.yaml → Parses CLI → Auto-starts Ollama → Runs pipeline        │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         pipeline.py (Orchestrator)                           │
│  Creates stage engines → Concurrent Stages 2-4 → Sort → Stage 5 (top N)    │
└───────┬─────────┬─────────┬─────────┬─────────┬────────────────────────────┘
        │         │         │         │         │
        ▼         ▼         ▼         ▼         ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Stage 1 │ │ Stage 2 │ │ Stage 3 │ │ Stage 4 │ │ Stage 5 │
   │   JD    │ │ Resume  │ │Semantic │ │Scoring  │ │Explain- │
   │ Parsing │ │ Parsing │ │Matching │ │         │ │ability  │
   │ (once)  │ │(parallel)││(parallel)││(parallel)││(top N)  │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │            │           │            │           │
        ▼            ▼           ▼            ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ LiteLLM │ │ LiteLLM │ │Sentence │ │ Scoring │ │ LiteLLM │
   │  (LLM)  │ │+ Regex  │ │Transf.  │ │ Formula │ │or Rules │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

### Component Responsibilities

| Layer | Component | Role |
|-------|-----------|------|
| **Config** | `config.yaml` | All settings in one place (model, paths, weights, performance) |
| **Entry** | `run.py` | CLI interface, config loading, Ollama auto-start, display, JSON export |
| **Orchestration** | `pipeline.py` | Concurrent execution, stage coordination, result sorting |
| **Extraction** | `file_loader.py` | Reads PDF/DOCX/TXT into raw text (OCR, text boxes, hyperlinks) |
| **Stage 1** | `jd_understanding.py` | LLM-based JD parsing → structured requirements |
| **Stage 2** | `resume_understanding.py` | Regex baseline + LLM → structured candidate profile |
| **Stage 3** | `semantic_matching.py` | Embedding similarity across 6 dimensions |
| **Stage 4** | `scoring.py` | Weighted formula → qualification percentage |
| **Stage 5** | `explainability.py` | LLM reasoning (top N) + rule-based fallback (rest) |
| **Models** | `models.py` | Pydantic data models for all pipeline data |

---

## Execution Flow (Sequence of Calls)

### Phase 1: Startup & Configuration

```
main()                                          ← run.py entry point
  ├── parse_args()                              ← Parse CLI arguments
  ├── load_config(config_path)                  ← Load config.yaml
  ├── Merge: CLI overrides config values        ← Priority: CLI > config > defaults
  ├── setup_logging(debug)                      ← Configure log level
  ├── ensure_ollama_running(model)              ← Auto-start Ollama + pull model
  └── asyncio.run(run_matching(args))           ← Start async execution
```

### Phase 2: File Loading

```
run_matching(args)
  ├── load_jd(args)                             ← Load Job Description text
  │     └── file_loader.extract_text()          ← PyPDF2 → pdfplumber → OCR
  └── load_resumes(args)                        ← Load all resume files
        └── file_loader.load_files_from_directory()
              └── extract_text() × N            ← For each file
```

### Phase 3: Pipeline Execution (Concurrent)

```
pipeline.match(jd_text, resume_texts)
  │
  ├── STAGE 1: jd_understanding.extract(jd_text)     ← Runs ONCE
  │
  ├── STAGES 2-4: CONCURRENT (bounded by semaphore)  ← Runs in PARALLEL
  │     └── For each resume (up to `concurrency` at a time):
  │           ├── Stage 2: resume_understanding.extract()
  │           ├── Stage 3: semantic_matcher.match()
  │           └── Stage 4: scorer.score()
  │
  ├── SORT by qualification_percentage (descending)
  │
  └── STAGE 5: explainability (SEQUENTIAL, top N only)
        ├── Top N candidates → LLM explanation
        └── Remaining candidates → rule-based fallback (instant)
```

### Phase 4: Output

```
run_matching(args)  (continued)
  ├── Display summary table (Source File, Name, Exp, %, Skills, Action)
  ├── Display detailed reports (per candidate)
  └── Save JSON (metadata + candidates array with source_file mapping)
```

---

## File-by-File Summary

| File | Purpose | Called by | Calls |
|------|---------|-----------|-------|
| `run.py` | CLI, config, Ollama auto-start, display, JSON export | User | config.yaml, file_loader, pipeline |
| `config.yaml` | All settings (model, paths, weights, performance) | run.py | — |
| `file_loader.py` | PDF/DOCX/TXT text extraction (OCR, text boxes, hyperlinks) | run.py | PyPDF2, pdfplumber, pytesseract, python-docx |
| `pipeline.py` | Orchestrates stages 1-6 with concurrency | run.py | All stage modules |
| `jd_understanding.py` | Stage 1: LLM-based JD parsing | pipeline.py | litellm |
| `resume_understanding.py` | Stage 2: Regex + LLM resume parsing | pipeline.py | litellm, regex |
| `semantic_matching.py` | Stage 3: Embedding similarity (6 dimensions) | pipeline.py | sentence-transformers |
| `scoring.py` | Stage 4: Weighted scoring formula | pipeline.py | — (internal math) |
| `explainability.py` | Stage 5: LLM explanation + rule-based fallback | pipeline.py | litellm |
| `models.py` | Pydantic data models | All modules | — |

---

## Project Structure

```
AI-Resume-Matcher/
├── run.py                              ← Main entry point
├── config.yaml                         ← Configuration (model, paths, weights, performance)
├── requirements.txt                    ← Python dependencies
├── .gitignore                          ← Git ignore rules
├── README.md                           ← This file
├── resumes/                            ← Place candidate resumes here (PDF/DOCX/TXT)
│   └── .gitkeep
├── jd/                                 ← Place JD file here (PDF/DOCX/TXT)
│   └── .gitkeep
└── matching_engine/                    ← Core framework package
    ├── __init__.py
    ├── models.py                       ← Pydantic data models
    ├── file_loader.py                  ← Text extraction (PDF/DOCX/TXT, OCR, text boxes)
    ├── jd_understanding.py             ← Stage 1: LLM-based JD parsing
    ├── resume_understanding.py         ← Stage 2: Regex + LLM resume parsing
    ├── semantic_matching.py            ← Stage 3: Embedding similarity (6 dimensions)
    ├── scoring.py                      ← Stage 4: Weighted scoring formula
    ├── explainability.py               ← Stage 5: LLM explanation + rule-based fallback
    ├── pipeline.py                     ← Pipeline orchestrator (concurrent stages)
    └── example_usage.py                ← Demo script with sample data
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| JD shows 0 characters | Scanned/image-based PDF | Install OCR: `brew install tesseract poppler && pip install pytesseract pdf2image` |
| All scores identical (78.6%) | Ollama not running OR JD empty | Script auto-starts Ollama. Check if JD file has extractable text. |
| SSL certificate errors | Corporate proxy (Zscaler) | Handled automatically (script sets `LITELLM_LOCAL_MODEL_COST_MAP=True`) |
| Embedding model fails | SSL blocks HuggingFace | Framework auto-patches httpx SSL verification |
| LLM returns bad JSON | Model too small | Use `ollama/llama3` or larger; framework retries 3 times + has fallbacks |
| DOCX shows minimal text | Content in text boxes/tables | Framework extracts from paragraphs, tables, text boxes, hyperlinks, headers, footers |
| Name/email/phone missing | LLM failed | Regex baseline always extracts contact info as fallback |
| Ollama not installed | Binary not found | `brew install ollama` or https://ollama.com/download |
| Model not found | Not pulled yet | Script auto-pulls on first run, or: `ollama pull llama3` |
| Slow (100+ resumes) | Sequential processing | Use `--concurrency 5 --explain-top 10` |

---

## Example Output

### Terminal (Summary Table)

```
⏱  Pipeline completed in 95.3 seconds (5 resumes)

====================================================================================================================
RESULTS — Candidate Match Grid (sorted by % Qualified)
====================================================================================================================
#   Source File                     First Name   Middle   Last Name       Contact Number     Email                        Exp(Yrs)  % Match  Key Skills (Top 3)                  Action
--------------------------------------------------------------------------------------------------------------------
1   DevSecOps_MLOps_v3.docx         Kumar        S        Karpuram        +91-96864-88688    shootmail2kumar@gmail.com    17.0      59.7%    MLOps, Docker, Kubernetes...         👍 Consider for interview
2   Kumar_DevSecOps_MLOps_v1.pdf    Kumar        S        Karpuram        +91-96864-88688    shootmail2kumar@gmail.com    13.0      56.6%    Python, AWS, Terraform...            👍 Consider for interview
3   Jyothi Kancharla.docx           Jyothi       -        Kancharla       +91-7093455517     Jyothi.Kancharla@gmail.com   15.0      52.2%    Data Science, Python, NLP...         👍 Consider for interview
4   Arun Prasad Resume.pdf          Arun         -        Prasad          9900912302         arunprasad@gmail.com         17.0      50.9%    Java, Spring, Microservices...       ⚠️  May need additional screening
5   Resume_Sr.Engineering_Spec...   Kumar        S        K               +91-96864-88688    shootmail2kumar@gmail.com    13.0      32.0%    Selenium, Java, DevOps...            ⚠️  May need additional screening
====================================================================================================================
```

### JSON Output (`results.json`)

See [JSON Output (for UI)](#json-output-for-ui) section above for full structure.

---

## License

Internal use only.
