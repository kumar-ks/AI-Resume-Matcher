# AI Resume Matcher

AI-powered Resume to Job Description matching engine with a 6-stage pipeline. Uses LLM-based extraction, embedding-based semantic matching, and weighted scoring to rank candidates against a job description.

---

## Table of Contents

- [Quick Start](#quick-start)
- [CLI Options](#cli-options)
- [LLM Providers](#llm-providers)
- [Framework Architecture](#framework-architecture)
- [Execution Flow (Sequence of Calls)](#execution-flow-sequence-of-calls)
- [File-by-File Summary](#file-by-file-summary)
- [Pipeline Stages (Detailed)](#pipeline-stages-detailed)
- [Data Models](#data-models)
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

### 2. Place your files

```
AI-Resume-Matcher/
├── resumes/                ← Drop all candidate resumes here
│   ├── rohit_sharma.pdf
│   ├── anita_iyer.docx
│   └── vikram_reddy.txt
├── jd/                     ← Place the job description here
│   └── senior_backend.pdf
└── run.py
```

**Supported formats:** PDF, DOCX, TXT

### 3. Run

```bash
# Basic (uses Ollama local LLM)
python run.py --model ollama/llama3

# With specific JD file
python run.py --jd ./jd/senior_backend.pdf --model ollama/llama3

# With OpenAI GPT-4
python run.py --model gpt-4

# Show only top 5 candidates
python run.py --model ollama/llama3 --top 5

# Save results to JSON
python run.py --model ollama/llama3 --output results.json

# Enable debug logging (verbose internal state)
python run.py --model ollama/llama3 --debug
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--resumes` | `./resumes` | Folder containing resume files |
| `--jd` | (auto from `./jd/`) | Path to specific JD file |
| `--jd-dir` | `./jd` | Folder containing JD file(s) |
| `--model` | `ollama/llama2` | LLM model identifier (see providers below) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `--top` | all | Show only top N candidates |
| `--output` | none | Save results to JSON file |
| `--debug` | off | Enable DEBUG-level logging (shows LLM responses, scores, regex matches) |

---

## LLM Providers

Uses [LiteLLM](https://github.com/BerriAI/litellm) — set the appropriate env var for your provider:

| Provider | Model flag | Env var needed |
|----------|-----------|----------------|
| Ollama (local) | `--model ollama/llama3` | None (run `ollama serve`) |
| OpenAI | `--model gpt-4` | `OPENAI_API_KEY` |
| Anthropic | `--model anthropic/claude-3-sonnet-20240229` | `ANTHROPIC_API_KEY` |
| AWS Bedrock | `--model bedrock/anthropic.claude-3-sonnet` | AWS credentials |

---

## Framework Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              run.py (Entry Point)                            │
│  Parses CLI args → Loads files → Initializes pipeline → Displays results    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         pipeline.py (Orchestrator)                           │
│  Creates all stage engines → Runs stages 1-6 for each resume → Sorts       │
└───────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────────────┘
        │         │         │         │         │         │
        ▼         ▼         ▼         ▼         ▼         ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Stage 1 │ │ Stage 2 │ │ Stage 3 │ │ Stage 4 │ │ Stage 5 │ │ Stage 6 │
   │   JD    │ │ Resume  │ │Semantic │ │Scoring  │ │Explain- │ │ Output  │
   │ Under-  │ │ Under-  │ │Matching │ │         │ │ability  │ │(Assemble│
   │standing │ │standing │ │         │ │         │ │         │ │ Result) │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └─────────┘
        │            │           │            │           │
        ▼            ▼           ▼            ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ LiteLLM │ │ LiteLLM │ │Sentence │ │ Scoring │ │ LiteLLM │
   │  (LLM)  │ │+ Regex  │ │Transf.  │ │ Formula │ │  (LLM)  │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

### Component Responsibilities

| Layer | Component | Role |
|-------|-----------|------|
| **Entry** | `run.py` | CLI interface, file loading, result display |
| **Orchestration** | `pipeline.py` | Creates stage engines, runs pipeline, sorts results |
| **Extraction** | `file_loader.py` | Reads PDF/DOCX/TXT files into raw text |
| **Stage 1** | `jd_understanding.py` | LLM-based JD parsing → structured requirements |
| **Stage 2** | `resume_understanding.py` | Regex baseline + LLM parsing → structured profile |
| **Stage 3** | `semantic_matching.py` | Embedding similarity across 6 dimensions |
| **Stage 4** | `scoring.py` | Weighted formula → qualification percentage |
| **Stage 5** | `explainability.py` | LLM-generated reasoning + recommendation |
| **Stage 6** | `pipeline.py` | Assembles final MatchResult object |
| **Models** | `models.py` | Pydantic data models for all pipeline data |

---

## Execution Flow (Sequence of Calls)

This section documents the exact sequence of method calls from start to finish.

### Phase 1: Startup & File Loading

```
main()                                          ← run.py entry point
  ├── parse_args()                              ← Parse CLI arguments (--model, --debug, etc.)
  ├── setup_logging(debug)                      ← Configure log level (INFO or DEBUG)
  └── asyncio.run(run_matching(args))           ← Start async execution
        │
        ├── load_jd(args)                       ← Load Job Description text
        │     └── file_loader.extract_text()    ← Route to format-specific reader
        │           ├── _read_pdf()             ← PyPDF2 → pdfplumber → OCR fallback
        │           ├── _read_docx()            ← Hyperlinks → Text boxes → Paragraphs → Tables
        │           └── _read_txt()             ← Plain UTF-8 read
        │
        └── load_resumes(args)                  ← Load all resume files
              └── file_loader.load_files_from_directory()
                    └── extract_text() × N      ← Called for each resume file
```

### Phase 2: Pipeline Initialization

```
run_matching(args)
  └── MatchingPipeline(model, embedding_model)  ← pipeline.py constructor
        ├── JDUnderstanding(model, temperature)       ← Stage 1 engine
        ├── ResumeUnderstanding(model, temperature)   ← Stage 2 engine
        ├── SemanticMatcher(embedding_model)          ← Stage 3 engine
        │     └── SentenceTransformer.load()          ← Pre-loads embedding model
        ├── Scorer(weights)                           ← Stage 4 engine
        └── ExplainabilityEngine(model, temperature)  ← Stage 5 engine
```

### Phase 3: Pipeline Execution (per JD × all resumes)

```
pipeline.match(jd_text, resume_texts)
  │
  ├── STAGE 1: jd_understanding.extract(jd_text)
  │     ├── litellm.acompletion()               ← Send JD to LLM
  │     ├── _extract_json(response)             ← Parse JSON from LLM output
  │     │     └── _try_parse_json()             ← Fix trailing commas, truncation
  │     └── _parse_response(data)               ← Convert dict → JobDescription model
  │
  └── FOR EACH RESUME:
        │
        ├── STAGE 2: resume_understanding.extract(resume_text)
        │     ├── _extract_baseline(text)       ← Regex: email, phone, name, experience
        │     │     ├── _extract_name_from_text()    ← Scan first 5 lines
        │     │     └── _estimate_experience_from_text()  ← Pattern matching
        │     ├── _extract_via_llm(text)        ← LLM extraction (with retries)
        │     │     ├── litellm.acompletion()   ← Send resume to LLM
        │     │     ├── _extract_json(response) ← Parse JSON (markdown, braces, truncation)
        │     │     └── _parse_response(data)   ← Convert dict → ResumeProfile
        │     │           ├── _estimate_experience(work_experiences)
        │     │           └── _estimate_experience_from_text(raw_text)
        │     └── _merge_profiles(baseline, llm_profile)  ← LLM priority, baseline fills gaps
        │
        ├── STAGE 3: semantic_matcher.match(jd, resume)
        │     ├── _compute_contextual_similarity()  ← Full text vs full text
        │     ├── _compute_skill_relevance()        ← JD skills vs resume skills
        │     ├── _compute_role_alignment()         ← Title+responsibilities vs work history
        │     ├── _compute_domain_relevance()       ← Industry/domain comparison
        │     ├── _compute_technology_mapping()     ← Tech stack comparison
        │     └── _compute_experience_alignment()   ← Years/level comparison (rule-based)
        │           └── _cosine_similarity(text_a, text_b)  ← Core embedding comparison
        │                 └── SentenceTransformer.encode()  ← Generate embeddings
        │
        ├── STAGE 4: scorer.score(jd, resume, semantic_result)
        │     ├── _score_must_have_match()      ← % of must-have skills matched
        │     │     └── _fuzzy_skill_match()    ← Exact + substring + abbreviation matching
        │     ├── _score_experience_match()     ← Years alignment (within/under/over range)
        │     ├── _score_skills_depth()         ← Breadth (60%) + semantic relevance (40%)
        │     ├── _score_project_relevance()    ← Tech overlap between projects and JD
        │     ├── _score_recency()              ← How recent is the experience
        │     └── WEIGHTED SUM × 100           ← Final qualification percentage
        │
        ├── STAGE 5: explainability.explain(jd, resume, scoring, semantic)
        │     ├── _build_prompt()               ← Format prompt with all match data
        │     ├── litellm.acompletion()         ← Send to LLM for explanation
        │     ├── _extract_json(response)       ← Parse JSON response
        │     └── _parse_response(data)         ← Convert to ExplainabilityReport
        │     └── (on failure) _fallback_explanation()  ← Rule-based fallback
        │
        └── STAGE 6: Assemble MatchResult       ← Combine all stage outputs
```

### Phase 4: Output & Display

```
run_matching(args)  (continued)
  ├── Sort results by qualification_percentage (descending)
  ├── Display results table (Name, Exp, Match %, Recommendation)
  ├── Display top candidate detail (strengths, gaps, reasoning)
  └── (optional) Save to JSON file
```

---

## File-by-File Summary

### `run.py` — Main Entry Point

| Aspect | Detail |
|--------|--------|
| **Purpose** | CLI interface, file loading, pipeline invocation, result display |
| **Called by** | User (command line) |
| **Calls** | `file_loader.extract_text()`, `file_loader.load_files_from_directory()`, `MatchingPipeline.match()` |
| **Key functions** | `main()`, `setup_logging()`, `parse_args()`, `load_jd()`, `load_resumes()`, `run_matching()` |
| **Notable** | Configures logging (INFO default, DEBUG with `--debug`), suppresses noisy third-party loggers |

### `matching_engine/file_loader.py` — Text Extraction

| Aspect | Detail |
|--------|--------|
| **Purpose** | Extract raw text from PDF, DOCX, and TXT files |
| **Called by** | `run.py` (load_jd, load_resumes) |
| **Calls** | PyPDF2, pdfplumber, pytesseract/pdf2image (OCR), python-docx |
| **Key functions** | `extract_text()`, `load_files_from_directory()`, `_read_pdf()`, `_read_docx()`, `_ocr_pdf()` |
| **Notable** | Multi-strategy PDF reading (PyPDF2 → pdfplumber → OCR). DOCX extraction covers paragraphs, tables, text boxes, hyperlinks, headers, and footers. |

### `matching_engine/pipeline.py` — Pipeline Orchestrator

| Aspect | Detail |
|--------|--------|
| **Purpose** | Creates all stage engines, runs the 6-stage pipeline, sorts results |
| **Called by** | `run.py` → `run_matching()` |
| **Calls** | All 5 stage modules (jd_understanding, resume_understanding, semantic_matching, scoring, explainability) |
| **Key class** | `MatchingPipeline` |
| **Key methods** | `match()`, `match_single()`, `_process_single_resume()`, `match_with_parsed_inputs()` |
| **Notable** | Stage 1 runs once per JD; Stages 2-6 run once per resume |

### `matching_engine/jd_understanding.py` — Stage 1: JD Parsing

| Aspect | Detail |
|--------|--------|
| **Purpose** | Extract structured requirements from raw JD text using LLM |
| **Called by** | `pipeline.py` → `match()` |
| **Calls** | `litellm.acompletion()` (LLM API) |
| **Key class** | `JDUnderstanding` |
| **Key methods** | `extract()`, `_extract_json()`, `_try_parse_json()`, `_parse_response()`, `_fallback_extraction()` |
| **Output** | `JobDescription` model (title, must_have_skills, good_to_have_skills, experience_range, education, domain, certifications, responsibilities) |
| **Notable** | Falls back to empty model (preserving raw_text) if LLM fails — downstream stages can still use raw text for semantic matching |

### `matching_engine/resume_understanding.py` — Stage 2: Resume Parsing

| Aspect | Detail |
|--------|--------|
| **Purpose** | Extract structured candidate profile from raw resume text |
| **Called by** | `pipeline.py` → `_process_single_resume()` |
| **Calls** | `litellm.acompletion()` (LLM API), regex patterns |
| **Key class** | `ResumeUnderstanding` |
| **Key methods** | `extract()`, `_extract_baseline()`, `_extract_via_llm()`, `_merge_profiles()`, `_extract_json()`, `_parse_response()`, `_estimate_experience()`, `_estimate_experience_from_text()` |
| **Output** | `ResumeProfile` model (name, email, phone, skills, experience, work_history, projects, education, certifications) |
| **Notable** | Two-pass strategy: (1) Regex baseline always succeeds for contact info, (2) LLM provides rich data. Merge ensures no field is ever empty if data exists in the text. Retries LLM up to 3 times on failure. |

### `matching_engine/semantic_matching.py` — Stage 3: Embedding Similarity

| Aspect | Detail |
|--------|--------|
| **Purpose** | Compute embedding-based semantic similarity across 6 dimensions |
| **Called by** | `pipeline.py` → `_process_single_resume()` |
| **Calls** | `SentenceTransformer.encode()` (sentence-transformers library) |
| **Key class** | `SemanticMatcher` |
| **Key methods** | `match()`, `_compute_contextual_similarity()`, `_compute_skill_relevance()`, `_compute_role_alignment()`, `_compute_domain_relevance()`, `_compute_technology_mapping()`, `_compute_experience_alignment()`, `_cosine_similarity()` |
| **Output** | `SemanticMatchResult` model (6 scores, each 0.0-1.0) |
| **Notable** | Pre-loads embedding model at init. Handles corporate proxy SSL issues (Zscaler) by patching httpx. Sets HF_HUB_OFFLINE after load to prevent async HTTP issues. |

### `matching_engine/scoring.py` — Stage 4: Weighted Scoring

| Aspect | Detail |
|--------|--------|
| **Purpose** | Compute weighted qualification percentage from multiple dimensions |
| **Called by** | `pipeline.py` → `_process_single_resume()` |
| **Calls** | Internal scoring methods (no external dependencies) |
| **Key class** | `Scorer` |
| **Key methods** | `score()`, `_score_must_have_match()`, `_score_experience_match()`, `_score_skills_depth()`, `_score_project_relevance()`, `_score_recency()`, `_fuzzy_skill_match()` |
| **Output** | `ScoringBreakdown` model (5 dimension scores + final qualification_percentage) |
| **Notable** | Configurable weights (default: must_have 35%, experience 25%, depth 20%, projects 12%, recency 8%). Fuzzy skill matching handles abbreviations (k8s=kubernetes, js=javascript, etc.) |

### `matching_engine/explainability.py` — Stage 5: Explanation Generation

| Aspect | Detail |
|--------|--------|
| **Purpose** | Generate human-readable reasoning for the match score |
| **Called by** | `pipeline.py` → `_process_single_resume()` |
| **Calls** | `litellm.acompletion()` (LLM API) |
| **Key class** | `ExplainabilityEngine` |
| **Key methods** | `explain()`, `_build_prompt()`, `_extract_json()`, `_parse_response()`, `_fallback_explanation()` |
| **Output** | `ExplainabilityReport` model (reason_for_score, matched_strengths, missing_skills, improvement_areas, recommendation) |
| **Notable** | Has a deterministic rule-based fallback that generates explanations from scoring data when LLM fails. Recommendation tiers: Strong Fit (≥85%), Good Fit (≥70%), Partial Fit (≥50%), Weak Fit (<50%). |

### `matching_engine/models.py` — Data Models

| Aspect | Detail |
|--------|--------|
| **Purpose** | Pydantic data models for all pipeline data structures |
| **Called by** | All stage modules |
| **Key models** | `JobDescription`, `ResumeProfile`, `SemanticMatchResult`, `ScoringBreakdown`, `ExplainabilityReport`, `MatchResult`, `Skill`, `WorkExperience`, `Project` |
| **Notable** | All models use Pydantic v2 with sensible defaults. `ResumeProfile.full_name` is a computed property combining first/middle/last. |

---

## Pipeline Stages (Detailed)

### Stage 1: JD Understanding

```
Input:  Raw JD text (string from PDF/DOCX/TXT)
Output: JobDescription model
Method: LLM-based extraction via litellm
```

Extracts:
- Job title
- Must-have skills (with optional years_required)
- Good-to-have skills
- Experience range (min/max years)
- Education requirements
- Domain/industry
- Certifications
- Responsibilities
- Location
- Role level (entry/mid/senior/lead/principal)

### Stage 2: Resume Understanding

```
Input:  Raw resume text (string from PDF/DOCX/TXT)
Output: ResumeProfile model
Method: Regex baseline + LLM extraction + merge
```

**Two-pass strategy:**

| Pass | Method | Reliability | Data Quality |
|------|--------|-------------|--------------|
| Pass 1 (Baseline) | Regex patterns | Always succeeds | Basic (name, email, phone, experience) |
| Pass 2 (LLM) | litellm.acompletion() | May fail | Rich (skills, work history, projects, education) |
| Merge | Priority logic | Always succeeds | Best of both |

Merge rules:
- LLM data wins for rich fields (skills, work history, projects)
- Baseline fills gaps in contact info (email, phone, name, experience)
- If LLM fails entirely, baseline-only profile is returned

### Stage 3: Semantic Matching

```
Input:  JobDescription + ResumeProfile
Output: SemanticMatchResult (6 scores, each 0.0-1.0)
Method: Sentence embeddings + cosine similarity
```

| Dimension | What it compares | Score meaning |
|-----------|-----------------|---------------|
| `contextual_similarity` | Full JD text vs full resume text | Overall relevance |
| `skill_relevance` | JD skills list vs resume skills list | Skill alignment |
| `role_alignment` | JD title+responsibilities vs work history | Role fit |
| `domain_relevance` | JD industry vs resume domains | Industry match |
| `technology_mapping` | JD tech stack vs resume tech stack | Tech alignment |
| `experience_alignment` | JD years range vs candidate years | Experience fit |

### Stage 4: Scoring

```
Input:  JobDescription + ResumeProfile + SemanticMatchResult
Output: ScoringBreakdown (5 scores + final percentage)
Method: Weighted formula (configurable weights)
```

**Default weights:**

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| `must_have_match` | 35% | % of must-have skills the candidate has |
| `experience_match` | 25% | How well experience years align with requirements |
| `skills_depth` | 20% | Breadth (60%) + semantic relevance (40%) |
| `project_relevance` | 12% | Technology overlap between projects and JD |
| `recency_factor` | 8% | How recent the relevant experience is |

**Formula:** `qualification_% = (Σ dimension_score × weight) × 100`

### Stage 5: Explainability

```
Input:  JobDescription + ResumeProfile + ScoringBreakdown + SemanticMatchResult
Output: ExplainabilityReport
Method: LLM-generated reasoning (with rule-based fallback)
```

Generates:
- Reason for score (2-3 sentences)
- Matched strengths (3-5 items)
- Missing skills
- Improvement areas
- Recommendation tier

### Stage 6: Output

```
Input:  All stage outputs
Output: MatchResult (final structured result per candidate)
Method: Assembly (no computation)
```

Combines candidate profile, scores, explanations, and recommendations into a single `MatchResult` object. Results are sorted by `qualification_percentage` (highest first).

---

## Data Models

```
MatchResult
├── candidate: ResumeProfile
│     ├── first_name, last_name, email, phone, location
│     ├── career_summary, skills[], certifications[]
│     ├── total_experience_years
│     ├── work_experiences[]: WorkExperience
│     │     ├── company, title, start_year, end_year, is_current
│     │     ├── technologies[], responsibilities[]
│     │     └── domain, duration_months
│     ├── projects[]: Project
│     │     ├── name, description, technologies[]
│     │     └── duration_months, role
│     ├── education[], domain_expertise[]
│     └── raw_text
├── job_description: JobDescription
│     ├── title, location, role_level
│     ├── must_have_skills[]: Skill (name, category, years_required)
│     ├── good_to_have_skills[]: Skill
│     ├── experience_range_min, experience_range_max
│     ├── education[], certifications[], responsibilities[]
│     ├── domain_industry[]
│     └── raw_text
├── qualification_percentage: float (0-100)
├── semantic_scores: SemanticMatchResult (6 floats)
├── scoring_breakdown: ScoringBreakdown (5 floats + percentage)
├── explainability: ExplainabilityReport
├── key_strengths[], missing_skills[]
├── reasoning: str
└── recommendation: str
```

---

## Project Structure

```
AI-Resume-Matcher/
├── run.py                              ← Main entry point (CLI, display, JSON output)
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
├── resumes/                            ← Place candidate resume files here
│   ├── .gitkeep
│   └── (your PDF/DOCX/TXT files)
├── jd/                                 ← Place job description file here
│   ├── .gitkeep
│   └── (your JD PDF/DOCX/TXT file)
└── matching_engine/                    ← Core framework package
    ├── __init__.py                     ← Package marker
    ├── models.py                       ← Pydantic data models (all pipeline data structures)
    ├── file_loader.py                  ← Text extraction (PDF/DOCX/TXT, OCR, text boxes)
    ├── jd_understanding.py             ← Stage 1: LLM-based JD parsing
    ├── resume_understanding.py         ← Stage 2: Regex + LLM resume parsing
    ├── semantic_matching.py            ← Stage 3: Embedding-based similarity (6 dimensions)
    ├── scoring.py                      ← Stage 4: Weighted scoring formula
    ├── explainability.py               ← Stage 5: LLM explanation + rule-based fallback
    ├── pipeline.py                     ← Pipeline orchestrator (creates engines, runs stages)
    └── example_usage.py                ← Demo script with hardcoded sample data
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| JD shows 0 characters | Scanned/image-based PDF | Install OCR: `brew install tesseract poppler && pip install pytesseract pdf2image` |
| All scores are identical | JD text is empty (extraction failed) | Check JD file format, convert to DOCX/TXT if needed |
| SSL certificate errors | Corporate proxy (Zscaler) | Set `export LITELLM_LOCAL_MODEL_COST_MAP=True` |
| Embedding model fails | SSL blocks HuggingFace download | The framework auto-patches httpx SSL; if still failing, download model manually |
| LLM returns empty/bad JSON | Model too small for structured output | Use `ollama/llama3` or larger; the framework retries 3 times automatically |
| DOCX shows minimal text | Content in text boxes/tables | Fixed: the framework extracts from paragraphs, tables, text boxes, hyperlinks, headers, and footers |
| Name/email/phone missing | LLM failed to extract | Fixed: regex baseline always extracts contact info as fallback |

---

## Example Output

```
======================================================================
AI RESUME MATCHER
======================================================================
  JD loaded: 5689 characters
  Resumes found: 5
  Model: ollama/llama3
  Embeddings: all-MiniLM-L6-v2
  Debug mode: OFF
======================================================================

======================================================================
RESULTS — Candidate Match Grid (sorted by Match %)
======================================================================
#    Name                      Exp      Match %    Recommendation
----------------------------------------------------------------------
1    Kumar S Karpuram          17.0y    59.7%      Good Fit - Consider for interview
2    Kumar S Karpuram          13.0y    56.6%      Good Fit - Consider for interview
3    Jyothi Kancharla          15.0y    52.2%      Good Fit - Consider for interview
4    Arun Prasad Sridharan     17.0y    50.9%      Good Fit - Consider for interview
5    Kumar S Karpuram          13.0y    32.0%      Partial Fit - May need additional screening

======================================================================
TOP CANDIDATE DETAIL
======================================================================
  Name:          Kumar S Karpuram
  Email:         shootmail2kumar@gmail.com
  Phone:         +91-96864-88688
  Experience:    17.0 years
  Match Score:   59.7%

  AI Reasoning:
    Strong match in MLOps, DevSecOps, and cloud platform skills...

  Matched Strengths:
    + Python expertise with ML libraries
    + Kubernetes and Docker containerization
    + AWS cloud platform experience

  Missing / Gap Areas:
    - Limited data science project experience

  Recommendation: Good Fit - Consider for interview
```

---

## License

Internal use only.
