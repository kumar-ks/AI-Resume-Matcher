# AI Resume Matcher — Setup, Run & Testing Guide

Complete guide to setting up and running the AI Resume Matcher framework with
LangChain, LangFuse, Bi-Frost, and LangGraph integrations.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Database Setup](#3-database-setup)
4. [Configuration](#4-configuration)
5. [Running the System](#5-running-the-system)
6. [Testing Each Integration](#6-testing-each-integration)
7. [API Usage](#7-api-usage)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### System Requirements

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| Docker | 20+ | PostgreSQL + pgvector |
| Ollama | Latest | Local LLM inference |
| pip | Latest | Package management |

### macOS Setup

```bash
# Install Docker Desktop
# https://www.docker.com/products/docker-desktop/

# Install Ollama
brew install ollama

# Pull the default model
ollama pull llama3
```

### Ubuntu Setup

```bash
# Docker
sudo apt install docker.io docker-compose

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

---

## 2. Installation

```bash
# Clone the repository
git clone <repo-url> AI-Resume-Matcher
cd AI-Resume-Matcher

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python3 -c "
from matching_engine.pipeline import MatchingPipeline
from matching_engine.graph import run_matching_graph
from matching_engine.chains.factory import get_llm_for_stage
from matching_engine.observability import get_langfuse
print('All imports OK')
"
```

---

## 3. Database Setup

### Start PostgreSQL + pgvector

```bash
docker-compose up -d
```

This starts a PostgreSQL 16 container with pgvector extension on port 5432.

### Verify Database Connection

```bash
docker exec -it resume_matcher_db psql -U matcher -d resume_matcher -c "SELECT 1;"
```

### Check Tables (created automatically on first run)

```bash
docker exec -it resume_matcher_db psql -U matcher -d resume_matcher -c "\dt"
```

Expected tables after first API call:
- `resume_profiles` — structured candidate data
- `resume_embeddings` — multi-field vector embeddings
- `match_results` — matching results for UI consumption

---

## 4. Configuration

### 4.1 Environment Variables

Copy the example and fill in your values:

```bash
cp .env.example .env
```

#### Minimum Required (.env)

```bash
# API authentication (generate your own)
AI_MATCHER_API_KEYS=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Database (default works with docker-compose)
DATABASE_URL=postgresql://matcher:matcher_secret@localhost:5432/resume_matcher
```

#### Optional: LangFuse Observability

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-your-key
LANGFUSE_SECRET_KEY=sk-lf-your-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

#### Optional: Bi-Frost Gateway

```bash
BIFROST_BASE_URL=https://bifrost-gateway.internal.bt.com/v1
BIFROST_API_KEY=bf-your-key
BIFROST_DEFAULT_MODEL=anthropic/claude-3-sonnet
BIFROST_PROJECT_ID=ai-resume-matcher
BIFROST_COST_CENTER=CC-1234
BIFROST_USE_CASE=recruitment-automation
```

### 4.2 config.yaml

Key settings to review:

```yaml
# LLM model (local Ollama or cloud)
model: "ollama/llama3"

# Performance
concurrency: 3          # Parallel resume processing
explain_top: null       # null = explain all, 5 = only top 5

# Pipeline mode
pipeline_mode: "linear" # or "graph" for LangGraph execution

# Bi-Frost
bifrost:
  enabled: false        # Set true when gateway is configured
  pii_routing: "default"
```

---

## 5. Running the System

### 5.1 Start Ollama (if using local models)

```bash
ollama serve
# In another terminal, verify:
ollama list
```

### 5.2 CLI Mode (Development/Testing)

#### Ingest Resumes

```bash
# Place resume files in ./resumes/ directory
# Place JD file in ./jd/ directory
python run.py --ingest --client-id ACME_CORP --job-id JOB-001
```

#### Match Against JD

```bash
python run.py --match --client-id ACME_CORP --job-id JOB-001
```

#### Check Database Status

```bash
python run.py --db-status
```

### 5.3 API Mode (Production)

```bash
# Start the FastAPI server
export AI_MATCHER_API_KEYS="your-secure-key-here"
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Swagger docs available at:
# http://localhost:8000/docs
```

### 5.4 Graph Mode (LangGraph Pipeline)

```python
# In Python code or a script:
import asyncio
from matching_engine.graph import run_matching_graph

async def main():
    results = await run_matching_graph(
        jd_text="Senior Python Developer...",
        resume_texts=["John Doe resume...", "Jane Smith resume..."],
        client_id="ACME",
        job_id="JOB-001",
        model="ollama/llama3",
    )
    for r in results:
        print(f"{r.candidate.full_name}: {r.qualification_percentage}%")

asyncio.run(main())
```

---

## 6. Testing Each Integration

### 6.1 Test LangChain (Structured Output)

Verify that LangChain chains produce valid Pydantic models:

```python
import asyncio
from matching_engine.chains.jd_chain import create_jd_chain

async def test_jd_chain():
    chain = create_jd_chain(model="ollama/llama3", temperature=0.1)

    result = await chain.ainvoke({
        "jd_text": """
        Senior Python Developer - London
        Requirements: 5+ years Python, AWS, Docker, Kubernetes.
        Nice to have: Terraform, CI/CD.
        """
    })

    print(f"Title: {result.title}")
    print(f"Must-have skills: {[s.name for s in result.must_have_skills]}")
    print(f"Experience: {result.experience_range_min}-{result.experience_range_max} years")
    assert result.title != ""
    assert len(result.must_have_skills) > 0
    print("JD chain test PASSED")

asyncio.run(test_jd_chain())
```

```python
import asyncio
from matching_engine.chains.resume_chain import create_resume_chain

async def test_resume_chain():
    chain = create_resume_chain(model="ollama/llama3", temperature=0.1)

    result = await chain.ainvoke({
        "resume_text": """
        John Smith
        john@email.com | +44 7700 900000
        Senior Python Developer with 8 years experience.
        Skills: Python, AWS, Docker, Kubernetes, Terraform
        Experience:
        - TechCorp (2020-Present): Lead Developer, Python microservices
        - StartupXYZ (2016-2020): Backend Developer, Django/Flask
        """
    })

    print(f"Name: {result.first_name} {result.last_name}")
    print(f"Skills: {result.skills}")
    print(f"Experience: {result.total_experience_years} years")
    assert result.first_name != ""
    assert len(result.skills) > 0
    print("Resume chain test PASSED")

asyncio.run(test_resume_chain())
```

### 6.2 Test LangFuse (Observability)

#### Step 1: Set LangFuse credentials

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

#### Step 2: Run a pipeline and check traces

```python
import asyncio
from matching_engine.graph import run_matching_graph

async def test_langfuse():
    results = await run_matching_graph(
        jd_text="Python Developer, 5 years experience required",
        resume_texts=["John Smith, 8 years Python, AWS, Docker"],
        client_id="TEST_CLIENT",
        job_id="TEST_JOB",
    )
    print(f"Results: {len(results)}")
    # Check LangFuse dashboard for traces tagged with client:TEST_CLIENT

asyncio.run(test_langfuse())
```

#### Step 3: Verify in LangFuse Dashboard

Navigate to your LangFuse project dashboard. You should see:
- A trace named `match-pipeline` or `api-ingest-and-match`
- Child spans for each stage (jd-extraction-chain, resume-extraction-chain, etc.)
- Token usage and latency metrics per generation
- Tags: `client:TEST_CLIENT`, `job:TEST_JOB`

#### Test Graceful Degradation (no credentials)

```bash
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
python3 -c "
from matching_engine.observability import get_langfuse
assert get_langfuse() is None
print('LangFuse gracefully disabled (no credentials)')
"
```

### 6.3 Test Bi-Frost Gateway (PII Routing)

#### Test PII Routing Logic (no actual gateway needed)

```bash
# Set env to simulate gateway configured
export BIFROST_BASE_URL=https://bifrost.test/v1
export BIFROST_API_KEY=test-key
export BIFROST_PROJECT_ID=ai-resume-matcher

python3 -c "
from matching_engine.chains.factory import get_llm_for_stage, is_bifrost_enabled
assert is_bifrost_enabled(), 'Should be enabled'

# JD goes through gateway (no PII)
jd_llm = get_llm_for_stage('jd', model='ollama/llama3')
assert type(jd_llm).__name__ == 'BifrostGatewayModel'
print(f'JD: {type(jd_llm).__name__} (gateway)')

# Resume stays local (PII protection)
resume_llm = get_llm_for_stage('resume', model='ollama/llama3')
assert type(resume_llm).__name__ == 'ChatLiteLLMModel'
print(f'Resume: {type(resume_llm).__name__} (local)')

# Explain goes through gateway
explain_llm = get_llm_for_stage('explain', model='ollama/llama3')
assert type(explain_llm).__name__ == 'BifrostGatewayModel'
print(f'Explain: {type(explain_llm).__name__} (gateway)')

print('PII routing test PASSED')
"
```

#### Test Override: Route All Through Gateway

```bash
export BIFROST_PII_STAGES=all

python3 -c "
from matching_engine.chains.factory import get_llm_for_stage
resume_llm = get_llm_for_stage('resume', model='ollama/llama3')
assert type(resume_llm).__name__ == 'BifrostGatewayModel'
print('All stages routed through gateway')
"
```

#### Test Gateway Headers

```python
from matching_engine.chains.gateway import BifrostGatewayModel

gw = BifrostGatewayModel(
    model="anthropic/claude-3-sonnet",
    gateway_url="https://bifrost.internal.bt.com/v1",
    gateway_api_key="bf-key",
    project_id="ai-resume-matcher",
    cost_center="CC-1234",
    use_case="recruitment-automation",
)

kwargs = gw._get_gateway_kwargs()
assert kwargs["api_base"] == "https://bifrost.internal.bt.com/v1"
assert kwargs["extra_headers"]["X-Project-Id"] == "ai-resume-matcher"
assert kwargs["extra_headers"]["X-Cost-Center"] == "CC-1234"
assert kwargs["extra_headers"]["X-Use-Case"] == "recruitment-automation"
print(f"Gateway headers: {kwargs['extra_headers']}")
print("Gateway headers test PASSED")
```

### 6.4 Test LangGraph (Stateful Pipeline)

#### Test Graph Compilation

```python
from matching_engine.graph import build_matching_graph, compile_graph

graph = build_matching_graph()
print(f"Nodes: {list(graph.nodes.keys())}")
# Expected: ['parse_jd', 'extract_resumes', 'score_resumes',
#            'explain_results', 'skip_explain', 'assemble_output']

compiled = compile_graph()
print(f"Compiled: {type(compiled).__name__}")
print("Graph compilation test PASSED")
```

#### Test Full Graph Execution (requires Ollama running)

```python
import asyncio
from matching_engine.graph import run_matching_graph

async def test_graph():
    results = await run_matching_graph(
        jd_text="""
        Senior Python Developer - 5+ years experience.
        Must have: Python, AWS, Docker.
        Nice to have: Kubernetes, Terraform.
        """,
        resume_texts=[
            """
            Jane Smith | jane@email.com
            8 years Python, AWS certified.
            Skills: Python, AWS, Docker, Kubernetes, Terraform
            TechCorp 2018-Present: Lead Developer
            """,
        ],
        model="ollama/llama3",
        client_id="TEST",
        job_id="GRAPH-TEST",
        explain_top_n=1,
    )

    assert len(results) == 1
    r = results[0]
    print(f"Candidate: {r.candidate.full_name}")
    print(f"Score: {r.qualification_percentage}%")
    print(f"Recommendation: {r.recommendation}")
    print("Graph execution test PASSED")

asyncio.run(test_graph())
```

#### Test Conditional Routing (Skip Explanations)

```python
import asyncio
from matching_engine.graph import run_matching_graph

async def test_skip_explain():
    results = await run_matching_graph(
        jd_text="Python Developer",
        resume_texts=["John, 5 years Python"],
        model="ollama/llama3",
        explain_top_n=0,  # Skip all LLM explanations
    )
    # Should use rule-based explanations only
    assert len(results) == 1
    print(f"Recommendation: {r.recommendation}")
    print("Skip-explain routing test PASSED")

asyncio.run(test_skip_explain())
```

---

## 7. API Usage

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "service": "ai-resume-matcher"}
```

### Ingest Resumes + Match

```bash
API_KEY="your-api-key"

curl -X POST http://localhost:8000/api/ingest \
  -H "X-API-Key: $API_KEY" \
  -F "client_id=ACME_CORP" \
  -F "job_id=JOB-001" \
  -F "jd_file=@./jd/job_description.docx" \
  -F "files=@./resumes/candidate1.pdf" \
  -F "files=@./resumes/candidate2.docx" \
  -F "explain=true"
```

### Check Results

```bash
curl "http://localhost:8000/api/status?client_id=ACME_CORP&job_id=JOB-001" \
  -H "X-API-Key: $API_KEY"
```

### Match Only (existing profiles)

```bash
curl -X POST http://localhost:8000/api/match \
  -H "X-API-Key: $API_KEY" \
  -F "client_id=ACME_CORP" \
  -F "job_id=JOB-002" \
  -F "jd_file=@./jd/new_role.docx"
```

### Mark Results as Delivered

```bash
curl -X POST http://localhost:8000/api/deliver \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "ACME_CORP",
    "job_id": "JOB-001",
    "result_ids": ["uuid-1", "uuid-2"]
  }'
```

---

## 8. Troubleshooting

### Ollama Not Running

```
Error: Connection refused on port 11434
```

**Fix:** Start Ollama with `ollama serve` or the API server auto-starts it.

### LiteLLM Model Cost Map Timeout

```
LiteLLM: Failed to fetch remote model cost map
```

**Fix:** Set `LITELLM_LOCAL_MODEL_COST_MAP=True` in your environment. This is cosmetic — the system falls back to local cost data automatically.

### Database Connection Failed

```
Error: connection to server at "localhost", port 5432 failed
```

**Fix:** Ensure Docker is running: `docker-compose up -d`

### LangFuse Traces Not Appearing

1. Verify credentials: `echo $LANGFUSE_PUBLIC_KEY`
2. Check host URL matches your project region
3. Traces are batched — wait 5-10 seconds for them to appear
4. Check `logs/api.log` for "LangFuse client initialized" message

### Bi-Frost Gateway Connection Failed

1. Verify VPN/network access to `BIFROST_BASE_URL`
2. Check API key validity
3. System gracefully falls back to local model if gateway is unreachable
4. Set `BIFROST_PII_STAGES=none` to disable gateway routing temporarily

### Sentence-Transformers Model Loading Slow

First-time startup downloads the `all-MiniLM-L6-v2` model (~80MB). Subsequent starts use the cached model. To pre-download:

```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Graph Pipeline Crash Recovery

If the graph pipeline crashes mid-execution, re-invoke with the same `thread_id`:

```python
# First run (crashes at score_resumes)
results = await run_matching_graph(..., thread_id="job-001-run-1")

# Resume from last checkpoint
results = await run_matching_graph(..., thread_id="job-001-run-1")
# Skips already-completed nodes (parse_jd, extract_resumes)
```

---

## Architecture After All Integrations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI Resume Matcher                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FastAPI Server (api/server.py)                                         │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LangGraph StateGraph (matching_engine/graph.py)                 │    │
│  │    START → parse_jd → extract_resumes → score → explain → END   │    │
│  │    [checkpointed state] [conditional routing]                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LangChain Chains (matching_engine/chains/)                      │    │
│  │    prompt → ChatLiteLLMModel → PydanticOutputParser (with retry) │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                 │
│       ▼                                                                 │
│  ┌───────────────────────┐    ┌────────────────────────────────────┐    │
│  │  LLM Factory          │    │  LangFuse Observability            │    │
│  │  (chains/factory.py)  │    │  (observability.py)                │    │
│  │                       │    │                                    │    │
│  │  PII routing:         │    │  Traces → Spans → Generations     │    │
│  │  JD → Gateway         │    │  Token usage, cost, latency       │    │
│  │  Resume → Local       │    │  Score tracking per candidate     │    │
│  │  Explain → Gateway    │    │                                    │    │
│  └───────────┬───────────┘    └────────────────────────────────────┘    │
│              │                                                           │
│    ┌─────────┼─────────────┐                                            │
│    ▼                       ▼                                            │
│  Bi-Frost Gateway       Local LiteLLM                                   │
│  (enterprise)           (Ollama/Cloud)                                  │
│                                                                         │
│  PostgreSQL + pgvector (profiles, embeddings, results, checkpoints)      │
└─────────────────────────────────────────────────────────────────────────┘
```
