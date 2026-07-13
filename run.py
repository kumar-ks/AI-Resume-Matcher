"""
AI Resume Matcher - Main Entry Point
=====================================

This is the orchestrator script that ties together the entire matching pipeline.

EXECUTION FLOW:
    1. main() → Parses CLI arguments, calls asyncio.run(run_matching(args))
    2. run_matching(args):
        a. load_jd(args) → Calls file_loader.extract_text() or load_files_from_directory()
        b. load_resumes(args) → Calls file_loader.load_files_from_directory()
        c. MatchingPipeline(...) → Initializes all 5 stage engines
        d. pipeline.match(jd_text, resume_texts) → Runs the 6-stage pipeline
        e. Display results in terminal
        f. Optionally save to JSON

USAGE:
    python run.py --resumes ./resumes --jd ./jd/backend_developer.pdf
    python run.py --resumes ./resumes --jd ./jd/job_description.txt --model ollama/llama3
    python run.py --resumes ./resumes --jd-dir ./jd --model gpt-4 --debug

DEPENDENCIES:
    - matching_engine.file_loader: Extracts text from PDF/DOCX/TXT files
    - matching_engine.pipeline: Orchestrates the 6-stage matching pipeline
"""

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from matching_engine.file_loader import extract_text, load_files_from_directory
from matching_engine.llm_client import LLMClient, validate_llm_access, estimate_token_usage
from matching_engine.pipeline import MatchingPipeline
from matching_engine.template_renderer import render_top_candidates
from matching_engine.database import ProfileDatabase
from matching_engine.vector_store import VectorStore
from matching_engine.scanner import scan_and_ingest

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG FILE PATH
# Default: config.yaml in the project root. Override with --config flag.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# Default: INFO level. Use --debug flag to enable DEBUG level for all modules.
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False, mode: str = "ingest") -> None:
    """
    Configure logging for the entire application.

    Outputs to:
        - Console (same as before)
        - File: logs/ingest.log OR logs/match.log (based on mode)

    File rotation: Daily rollover (midnight), keeping 30 days of history.
    Format: Full date + timestamp in both console and file.
    Also captures print() output into the log file.

    Called by: main()

    Args:
        debug: If True, sets log level to DEBUG for verbose output.
        mode: "ingest" or "match" — determines which log file to write to.
    """
    from logging.handlers import TimedRotatingFileHandler

    log_level = logging.DEBUG if debug else logging.INFO

    # Ensure logs directory exists
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Choose log file based on mode
    log_file = log_dir / f"{mode}.log"

    # Full date+time format for both console and file
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers (avoid duplicates on re-init)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # File handler with daily rollover (rotates at midnight, keeps 30 days)
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    file_handler.suffix = "%Y-%m-%d"  # Rolled files: ingest.log.2026-07-06
    root_logger.addHandler(file_handler)

    # ── Intercept print() → also write to log file ────────────────────────
    # This ensures all console output (print statements) also appears in the log.
    sys.stdout = _TeeWriter(sys.stdout, log_file)

    # Suppress noisy third-party loggers unless in debug mode
    if not debug:
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("numexpr").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)

    logger.info(f"Logging initialized: mode={mode}, file={log_file}, level={'DEBUG' if debug else 'INFO'}")


class _TeeWriter:
    """
    Intercepts sys.stdout to mirror all print() output into a log file.
    Ensures every console line also appears in the log with a timestamp.
    """

    def __init__(self, original_stdout, log_file: Path):
        self.original = original_stdout
        self.log_file = log_file

    def write(self, text: str):
        # Write to original stdout (console)
        self.original.write(text)
        # Also append to log file (skip empty lines)
        if text.strip():
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | PRINT   | stdout | {text.rstrip()}\n")

    def flush(self):
        self.original.flush()

    def fileno(self):
        return self.original.fileno()


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load configuration from YAML file.

    Called by: main()
    Returns: dict with all config values (or empty dict if file not found)

    The config file provides default values for all settings.
    CLI arguments override config file values (CLI takes priority).
    """
    if not config_path.exists():
        logger.debug(f"Config file not found at {config_path}, using defaults")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.debug(f"Loaded config from {config_path}: {list(config.keys())}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config file {config_path}: {e}")
        return {}


def ensure_ollama_running(model: str) -> None:
    """
    Check if Ollama is required and running. If not running, start it automatically.

    Called by: main() before pipeline execution (only for ollama/* models).

    Flow:
        1. Check if the model uses Ollama (starts with "ollama/")
        2. If not an Ollama model, skip (cloud APIs don't need Ollama)
        3. Check if 'ollama' binary is installed on the system
        4. Try connecting to Ollama server (ollama list)
        5. If not running, start 'ollama serve' in the background
        6. Wait up to 10 seconds for the server to become ready
        7. Verify the required model is available locally

    Args:
        model: The LLM model identifier (e.g., "ollama/llama3", "gpt-4")
    """
    # Only needed for Ollama models
    if not model.startswith("ollama/"):
        logger.debug(f"Model '{model}' is not Ollama-based, skipping Ollama check")
        return

    # Check if ollama binary exists
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print("\nERROR: Ollama is not installed.")
        print("  Install from: https://ollama.com/download")
        print("  Or use a cloud model: python run.py --model gpt-4")
        sys.exit(1)

    # Check if Ollama server is already running
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.debug("Ollama server is already running")
            # Verify the required model is available
            model_name = model.replace("ollama/", "")
            if model_name not in result.stdout:
                print(f"\n  WARNING: Model '{model_name}' not found locally.")
                print(f"  Pulling model (this may take a few minutes on first run)...")
                subprocess.run(["ollama", "pull", model_name], check=False)
            # Warm up the model (load into memory) to avoid cold-start timeouts
            _warmup_ollama_model(model_name)
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass

    # Ollama is not running — start it automatically
    print("  Starting Ollama server...")
    logger.info("Ollama not running, starting 'ollama serve' in background")

    # Start ollama serve as a background process (detached from this script)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for the server to become ready (up to 10 seconds)
    for i in range(10):
        time.sleep(1)
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                logger.info(f"Ollama server started successfully (took {i+1}s)")
                print(f"  Ollama server started (took {i+1}s)")

                # Check if model is available
                model_name = model.replace("ollama/", "")
                if model_name not in result.stdout:
                    print(f"  Pulling model '{model_name}' (first run may take a few minutes)...")
                    subprocess.run(["ollama", "pull", model_name], check=False)
                return
        except (subprocess.TimeoutExpired, Exception):
            continue

    # Failed to start after 10 seconds
    print("\n  ERROR: Could not start Ollama server after 10 seconds.")
    print("  Try starting it manually: ollama serve")
    sys.exit(1)


def _warmup_ollama_model(model_name: str) -> None:
    """
    Warm up an Ollama model by sending a tiny request to pre-load it into memory.

    Ollama unloads models from GPU/RAM after idle time. The first real request
    after unload takes 30-60s just to reload. This warmup forces the model to
    load AND sets keep_alive to prevent unloading during the pipeline run.

    Called by: ensure_ollama_running()
    """
    import httpx

    try:
        print(f"  Warming up model '{model_name}' (loading into memory)...")
        # Send a minimal generate request with keep_alive to prevent model unloading.
        # keep_alive="30m" keeps the model loaded for 30 minutes (enough for any pipeline run).
        with httpx.Client(verify=False, timeout=180.0) as client:
            response = client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": "30m",  # Keep model in memory for 30 minutes
                },
            )
            if response.status_code == 200:
                print(f"  ✓ Model '{model_name}' is warm and ready (keep_alive=30m)")
                logger.info(f"Ollama model '{model_name}' warmed up, keep_alive=30m")
            else:
                logger.warning(f"Warmup got status {response.status_code}, proceeding anyway")
    except Exception as e:
        # Non-fatal — pipeline will still work, just first call may be slow
        logger.warning(f"Model warmup failed (non-fatal): {e}")
        print(f"  ⚠️  Warmup skipped (first LLM call may be slow)")


def parse_args():
    """
    Parse command-line arguments.

    Called by: main()
    Returns: argparse.Namespace with all configuration options.
    """
    parser = argparse.ArgumentParser(
        description="AI Resume Matcher - Match resumes against a job description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --resumes ./resumes --jd ./jd/senior_backend.pdf
  python run.py --resumes ./resumes --jd ./jd/jd.txt --model ollama/llama3
  python run.py --resumes ./resumes --jd ./jd/jd.docx --top 5 --debug
  python run.py --resumes ./resumes --jd-dir ./jd --output results.json
  python run.py --config my_config.yaml

Folder structure:
  AI-Resume-Matcher/
  ├── config.yaml           ← Configuration (model, paths, weights)
  ├── resumes/              ← Place candidate resumes here
  │   ├── rohit_sharma.pdf
  │   ├── anita_iyer.docx
  │   └── vikram_reddy.txt
  ├── jd/                   ← Place job description here
  │   └── senior_backend_developer.pdf
  └── run.py
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--resumes",
        type=str,
        default="./resumes",
        help="Path to folder containing resume files (default: ./resumes)",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default=None,
        help="Path to JD file (PDF/DOCX/TXT). If a directory, uses the first file found.",
    )
    parser.add_argument(
        "--jd-dir",
        type=str,
        default="./jd",
        help="Path to folder containing JD file(s) (default: ./jd). Used if --jd not specified.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/llama2",
        help="LLM model to use (default: ollama/llama2). Options: ollama/llama3, gpt-4, etc.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for embeddings (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N candidates (default: show all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file (optional)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging (verbose output including LLM responses, scores, etc.)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of resumes to process in parallel (default: 3). "
             "Higher = faster but uses more memory. Set to 1 for sequential.",
    )
    parser.add_argument(
        "--explain-top",
        type=int,
        default=None,
        help="Only generate AI explanations for top N candidates (saves ~15s per skipped resume). "
             "Default: explain all. Recommended for large batches: --explain-top 10",
    )
    parser.add_argument(
        "--generate-doc",
        type=int,
        default=None,
        help="Generate formatted DOCX for top N candidates using the template. "
             "Output saved to rendered/ folder. Example: --generate-doc 3",
    )
    # ── Database & Scanner flags ──────────────────────────────────────────────
    parser.add_argument(
        "--ingest",
        action="store_true",
        default=False,
        help="Ingest resumes into the database (extract via LLM and store). "
             "Only processes new/modified files (skips already-ingested ones).",
    )
    parser.add_argument(
        "--match",
        action="store_true",
        default=False,
        help="Match against stored profiles in DB (fast, no re-extraction). "
             "Requires --ingest to have been run at least once.",
    )
    parser.add_argument(
        "--scan-mode",
        type=str,
        default=None,
        choices=["db_first", "folder_only", "db_only"],
        help="How to find candidates: "
             "db_first (default: DB + scan new files), "
             "folder_only (bypass DB, scan folder fresh), "
             "db_only (only use DB, fastest).",
    )
    parser.add_argument(
        "--db-status",
        action="store_true",
        default=False,
        help="Display database and vector store status, then exit.",
    )
    # ── Multi-tenant isolation flags ──────────────────────────────────────────
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Client identifier (NDA isolation boundary). REQUIRED for --ingest and --match. "
             "Resumes under one client-id can NEVER be shared with another client.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job opening identifier. REQUIRED for --ingest and --match. "
             "Resumes within the same client can be shared across job-ids.",
    )
    return parser.parse_args()


def load_jd(args) -> str:
    """
    Load Job Description text from file or directory.

    Called by: run_matching(args)
    Calls: file_loader.extract_text(), file_loader.load_files_from_directory()

    Resolution order:
        1. If --jd is a file path → load that file directly
        2. If --jd is a directory → load first supported file from it
        3. If --jd not specified → scan --jd-dir (default: ./jd/)

    Returns:
        Raw JD text as string

    Exits:
        If no JD file is found, prints error and exits with code 1.
    """
    if args.jd:
        jd_path = Path(args.jd)
        if jd_path.is_file():
            logger.info(f"Loading JD from file: {jd_path}")
            return extract_text(jd_path)
        elif jd_path.is_dir():
            # --jd points to a directory, load first file from it
            files = load_files_from_directory(jd_path)
            if files:
                logger.info(f"Using JD: {files[0]['filename']}")
                return files[0]["text"]
    else:
        # No --jd specified, scan the default JD directory
        jd_dir = Path(args.jd_dir)
        if jd_dir.exists():
            files = load_files_from_directory(jd_dir)
            if files:
                logger.info(f"Using JD: {files[0]['filename']}")
                return files[0]["text"]

    # No JD found — exit with helpful message
    print("\nERROR: No JD file found.")
    print("  Place a JD file in ./jd/ folder, or specify with --jd path/to/jd.pdf")
    sys.exit(1)


def load_resumes(args) -> list[dict]:
    """
    Load all resume files from the specified directory.

    Called by: run_matching(args)
    Calls: file_loader.load_files_from_directory()

    Returns:
        List of dicts, each with keys: 'filename', 'path', 'text'

    Exits:
        If directory doesn't exist or contains no supported files.
    """
    resumes_dir = Path(args.resumes)

    if not resumes_dir.exists():
        resumes_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated folder: {resumes_dir}/")
        print(f"  Place your resume files (PDF/DOCX/TXT) in this folder and run again.")
        sys.exit(1)

    files = load_files_from_directory(resumes_dir)

    if not files:
        print(f"\nERROR: No resume files found in {resumes_dir}/")
        print(f"  Supported formats: PDF, DOCX, TXT")
        print(f"  Place resume files there and run again.")
        sys.exit(1)

    return files


async def run_matching(args):
    """
    Execute the full matching pipeline and display results.

    Called by: main() via asyncio.run()
    Calls:
        - load_jd(args) → Get JD text
        - load_resumes(args) → Get resume texts
        - MatchingPipeline(...) → Initialize pipeline (creates all stage engines)
        - pipeline.match(jd_text, resume_texts) → Run 6-stage pipeline
        - Display results in terminal
        - Optionally save to JSON

    Pipeline stages (executed inside pipeline.match()):
        Stage 1: JD Understanding (LLM) → Extract skills, experience, requirements
        Stage 2: Resume Understanding (LLM + regex) → Extract profile data
        Stage 3: Semantic Matching (embeddings) → Compute similarity scores
        Stage 4: Scoring (weighted formula) → Calculate qualification %
        Stage 5: Explainability (LLM) → Generate human-readable reasoning
        Stage 6: Output → Assemble final MatchResult
    """
    # ── Step 0: Validate LLM access (token/API check) ────────────────────────
    # Makes a lightweight test call to verify the model is reachable before
    # processing all resumes. Fails over to backup model if primary is down.
    failover_model = getattr(args, "failover_model", None)
    print("\n  Validating LLM access...")
    validation = await validate_llm_access(args.model, failover_model)

    if validation["status"] == "failed":
        print(f"\n  ERROR: {validation['message']}")
        print("  Check your API keys, network connection, or Ollama server.")
        sys.exit(1)
    elif validation["status"] == "failover":
        print(f"  ⚠️  {validation['message']}")
        args.model = validation["model"]  # Switch to failover model for pipeline
    else:
        print(f"  ✓ {validation['message']}")

    # ── Step 1: Load input files ──────────────────────────────────────────────
    jd_text = load_jd(args)
    resume_files = load_resumes(args)

    # ── Step 2: Display configuration banner ──────────────────────────────────
    print("\n" + "=" * 70)
    print("AI RESUME MATCHER")
    print("=" * 70)
    print(f"  JD loaded: {len(jd_text)} characters")
    print(f"  Resumes found: {len(resume_files)}")
    print(f"  Model: {args.model}" + (" (failover)" if validation["status"] == "failover" else ""))
    print(f"  Failover: {failover_model or 'none'}")
    print(f"  Embeddings: {args.embedding_model}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Debug mode: {'ON' if args.debug else 'OFF'}")
    print("=" * 70)

    # ── Step 2.5: Token estimation (pre-flight budget check) ──────────────────
    # Estimates total tokens needed BEFORE running the pipeline.
    # For paid models: shows cost estimate and token budget.
    # For free models (Ollama): only checks context window limits.
    resume_texts = [f["text"] for f in resume_files]
    token_estimate = estimate_token_usage(
        jd_text=jd_text,
        resume_texts=resume_texts,
        model=args.model,
        explain_top_n=args.explain_top,
    )

    is_paid_model = token_estimate['estimated_cost_usd'] > 0

    if is_paid_model:
        # Show full token budget for paid models (cost matters)
        print(f"\n  Token Budget Estimate (paid model):")
        print(f"    Input tokens:  ~{token_estimate['total_input_tokens']:,}")
        print(f"    Output tokens: ~{token_estimate['total_output_tokens']:,}")
        print(f"    Total tokens:  ~{token_estimate['total_tokens']:,}")
        print(f"    LLM calls:     {token_estimate['num_llm_calls']}")
        print(f"    Context window: {token_estimate['context_window']:,} tokens")
        print(f"    💰 Estimated cost: ${token_estimate['estimated_cost_usd']:.4f} USD")

    # Display warnings (context window exceeded) — relevant for ALL models
    if token_estimate['warnings']:
        print(f"\n  ⚠️  Context Window Warnings:")
        for w in token_estimate['warnings']:
            print(f"    - {w}")

    if not token_estimate['sufficient']:
        print(f"\n  ERROR: Some inputs exceed the model's context window ({token_estimate['context_window']:,} tokens).")
        print(f"  Consider using a model with a larger context (e.g., gpt-4o, claude-3-sonnet)")
        print(f"  or reducing resume/JD size.")
        sys.exit(1)

    print()

    # ── Step 3: Initialize the matching pipeline ──────────────────────────────
    #   - JDUnderstanding (Stage 1)
    #   - ResumeUnderstanding (Stage 2)
    #   - SemanticMatcher (Stage 3) — also pre-loads the embedding model
    #   - Scorer (Stage 4)
    #   - ExplainabilityEngine (Stage 5)
    pipeline = MatchingPipeline(
        model=args.model,
        embedding_model=args.embedding_model,
        concurrency=args.concurrency,
        explain_top_n=args.explain_top,
    )

    # ── Step 4: Run the matching pipeline ─────────────────────────────────────
    # pipeline.match() executes stages 1-6 for each resume against the JD
    # (resume_texts already prepared in Step 2.5 for token estimation)
    results = await pipeline.match(jd_text=jd_text, resume_texts=resume_texts)

    # ── Step 5: Build filename mapping ──────────────────────────────────────────
    # Maps raw_text → {filename, path} for each resume.
    # Used for: (1) fallback display when name extraction fails,
    #           (2) tracking source file in JSON output for UI consumption.
    text_to_file = {f["text"]: {"filename": f["filename"], "path": f["path"]} for f in resume_files}

    # ── Step 6: Apply top-N filter if requested ───────────────────────────────
    if args.top:
        results = results[: args.top]

    # ── Step 7: Display results table ─────────────────────────────────────────
    # Format: First Name | Middle Name | Last Name | Contact Number | Email |
    #         Total Exp.(Years) | % Qualified against JD | Key Skills (Top) |
    #         Reasoning (Why this score) | Action
    print("\n" + "=" * 180)
    print("RESULTS — Candidate Match Grid (sorted by % Qualified)")
    print("=" * 180)

    # Table header
    header = (
        f"{'#':<3} "
        f"{'Source File':<30} "
        f"{'First Name':<12} "
        f"{'Middle':<8} "
        f"{'Last Name':<15} "
        f"{'Contact Number':<18} "
        f"{'Email':<28} "
        f"{'Exp(Yrs)':<9} "
        f"{'% Match':<8} "
        f"{'Key Skills (Top 3)':<35} "
        f"{'Action'}"
    )
    print(header)
    print("-" * 180)

    for i, result in enumerate(results, 1):
        c = result.candidate  # shorthand for candidate profile

        # Source file tracking
        file_info = text_to_file.get(c.raw_text, {"filename": f"unknown_{i}", "path": ""})
        source_file = file_info["filename"]
        # Truncate long filenames for display
        if len(source_file) > 28:
            source_file = source_file[:25] + "..."

        # Name fields (fall back to filename if extraction failed)
        first_name = c.first_name or "-"
        middle_name = c.middle_name or "-"
        last_name = c.last_name or "-"

        # Contact info
        phone = c.phone or "N/A"
        email = c.email or "N/A"

        # Experience
        exp = f"{c.total_experience_years}" if c.total_experience_years else "N/A"

        # Match percentage
        score = f"{result.qualification_percentage}%"

        # Key skills (top 3, comma-separated)
        top_skills = ", ".join(c.skills[:3]) if c.skills else "N/A"
        # Truncate if too long for display
        if len(top_skills) > 33:
            top_skills = top_skills[:30] + "..."

        # Action (recommendation)
        action = result.recommendation or "N/A"
        # Shorten common recommendation prefixes for table fit
        action = (
            action.replace("Strong Fit - ", "✅ ")
            .replace("Good Fit - ", "👍 ")
            .replace("Partial Fit - ", "⚠️  ")
            .replace("Weak Fit - ", "❌ ")
        )

        row = (
            f"{i:<3} "
            f"{source_file:<30} "
            f"{first_name:<12} "
            f"{middle_name:<8} "
            f"{last_name:<15} "
            f"{phone:<18} "
            f"{email:<28} "
            f"{exp:<9} "
            f"{score:<8} "
            f"{top_skills:<35} "
            f"{action}"
        )
        print(row)

    print("=" * 180)

    # ── Step 8: Display detailed view for each candidate ──────────────────────
    if results:
        print("\n" + "=" * 80)
        print("DETAILED CANDIDATE REPORTS")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            c = result.candidate
            file_info = text_to_file.get(c.raw_text, {"filename": "unknown", "path": ""})
            print(f"\n{'─' * 80}")
            print(f"  #{i} — {c.full_name or 'Unknown'}")
            print(f"{'─' * 80}")
            print(f"  Source File:   {file_info['filename']}")
            print(f"  File Path:     {file_info['path']}")
            print(f"  First Name:    {c.first_name or 'N/A'}")
            print(f"  Middle Name:   {c.middle_name or 'N/A'}")
            print(f"  Last Name:     {c.last_name or 'N/A'}")
            print(f"  Phone:         {c.phone or 'N/A'}")
            print(f"  Email:         {c.email or 'N/A'}")
            print(f"  Location:      {c.location or 'N/A'}")
            print(f"  Experience:    {c.total_experience_years or 'N/A'} years")
            print(f"  Match Score:   {result.qualification_percentage}%")
            print(f"\n  Key Skills:")
            for s in c.skills[:5]:
                print(f"    • {s}")
            print(f"\n  AI Reasoning:")
            print(f"    {result.reasoning or 'N/A'}")
            print(f"\n  Matched Strengths:")
            for s in result.key_strengths:
                print(f"    + {s}")
            print(f"\n  Missing / Gap Areas:")
            for s in result.missing_skills:
                print(f"    - {s}")
            print(f"\n  Action: {result.recommendation}")

    # ── Step 9: Save to JSON if --output specified ────────────────────────────
    # This JSON is designed for UI consumption. Each entry maps back to the
    # source resume file via 'source_file' and 'source_path' fields.
    if args.output:
        import json

        output_data = {
            "metadata": {
                "jd_file": str(Path(args.jd).name) if args.jd else "auto-detected from jd/",
                "jd_title": results[0].job_description.title if results else "",
                "total_candidates": len(results),
                "model_used": args.model,
                "embedding_model": args.embedding_model,
            },
            "candidates": [],
        }

        for rank, r in enumerate(results, 1):
            file_info = text_to_file.get(r.candidate.raw_text, {"filename": "unknown", "path": ""})
            output_data["candidates"].append({
                # ── Resume source tracking (for UI to link back to file) ──
                "rank": rank,
                "source_file": file_info["filename"],
                "source_path": file_info["path"],

                # ── Candidate profile ──
                "first_name": r.candidate.first_name or None,
                "middle_name": r.candidate.middle_name or None,
                "last_name": r.candidate.last_name or None,
                "full_name": r.candidate.full_name or None,
                "contact_number": r.candidate.phone or None,
                "email": r.candidate.email or None,
                "location": r.candidate.location or None,
                "total_experience_years": r.candidate.total_experience_years,

                # ── Match results ──
                "qualification_percentage": r.qualification_percentage,
                "action": r.recommendation,
                "reasoning": r.reasoning,

                # ── Skills & analysis ──
                "key_skills_top_5": r.candidate.skills[:5],
                "all_skills": r.candidate.skills,
                "matched_strengths": r.key_strengths,
                "missing_skills": r.missing_skills,

                # ── Scoring breakdown (for charts/visualizations in UI) ──
                "scoring_breakdown": {
                    "must_have_match": round(r.scoring_breakdown.must_have_match, 3),
                    "experience_match": round(r.scoring_breakdown.experience_match, 3),
                    "skills_depth": round(r.scoring_breakdown.skills_depth, 3),
                    "project_relevance": round(r.scoring_breakdown.project_relevance, 3),
                    "recency_factor": round(r.scoring_breakdown.recency_factor, 3),
                },

                # ── Work history (for detailed UI view) ──
                "work_experiences": [
                    {
                        "company": exp.company,
                        "title": exp.title,
                        "start_year": exp.start_year,
                        "end_year": exp.end_year,
                        "is_current": exp.is_current,
                        "technologies": exp.technologies,
                    }
                    for exp in r.candidate.work_experiences
                ],

                # ── Education & certifications ──
                "education": r.candidate.education,
                "certifications": r.candidate.certifications,
            })

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n\nResults saved to: {args.output}")
        print(f"  → {len(results)} candidates mapped to their source resume files")
        print(f"  → Use 'source_file' / 'source_path' fields to link UI back to resumes")

    # ── Step 10: Generate formatted DOCX if --generate-doc specified ──────────
    generate_count = getattr(args, "generate_doc", None)
    if generate_count:
        # Find the template file
        template_dir = Path("template")
        template_files = list(template_dir.glob("*.docx")) if template_dir.exists() else []

        if not template_files:
            print("\n  ⚠️  No DOCX template found in template/ folder. Skipping doc generation.")
        else:
            template_path = template_files[0]  # Use first template found
            output_dir = Path("rendered")

            print(f"\n  Generating formatted documents (top {generate_count})...")
            print(f"  Template: {template_path.name}")
            print(f"  Output folder: {output_dir}/")

            rendered = render_top_candidates(
                results=results,
                source_filenames=text_to_file,
                template_path=template_path,
                output_dir=output_dir,
                top_n=generate_count,
            )

            if rendered:
                print(f"\n  ✓ Generated {len(rendered)} formatted document(s) in '{output_dir}/'")


def main():
    """
    Application entry point.

    Flow:
        1. parse_args() → Parse CLI arguments
        2. load_config() → Load YAML config file
        3. Merge: CLI args override config values
        4. setup_logging() → Configure log levels
        5. ensure_ollama_running() → Auto-start Ollama if needed
        6. asyncio.run(run_matching()) → Execute pipeline

    Priority order (highest wins):
        CLI flags > config.yaml > built-in defaults
    """
    args = parse_args()

    # ── Load config file ──────────────────────────────────────────────────────
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    # ── Merge config into args (CLI takes priority over config) ───────────────
    # For each setting, use CLI value if explicitly provided, else config value.
    # argparse defaults are used only if neither CLI nor config provides a value.
    if config:
        # Model: CLI flag overrides config
        if args.model == "ollama/llama2" and config.get("model"):
            args.model = config["model"]
        # Failover model (only from config, no CLI flag for this)
        args.failover_model = config.get("failover_model")
        # Embedding model
        if args.embedding_model == "all-MiniLM-L6-v2" and config.get("embedding_model"):
            args.embedding_model = config["embedding_model"]
        # Paths
        if args.resumes == "./resumes" and config.get("resumes_dir"):
            args.resumes = config["resumes_dir"]
        if args.jd_dir == "./jd" and config.get("jd_dir"):
            args.jd_dir = config["jd_dir"]
        # Performance
        if args.concurrency == 3 and config.get("concurrency"):
            args.concurrency = config["concurrency"]
        if args.explain_top is None and config.get("explain_top"):
            args.explain_top = config["explain_top"]
        # Output
        if args.top is None and config.get("top_n"):
            args.top = config["top_n"]
        if args.output is None and config.get("output_file"):
            args.output = config["output_file"]
        # Debug
        if not args.debug and config.get("debug"):
            args.debug = config["debug"]
    else:
        args.failover_model = None

    # Determine log mode based on CLI flags
    if args.match:
        log_mode = "match"
    elif args.ingest:
        log_mode = "ingest"
    elif getattr(args, "scan_mode", None) in ("folder_only", "db_only"):
        log_mode = "match"
    else:
        log_mode = "ingest"

    setup_logging(debug=args.debug, mode=log_mode)
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Primary model: {args.model} | Failover: {args.failover_model or 'none'}")
    logger.debug(f"Effective settings: concurrency={args.concurrency}, explain_top={args.explain_top}")

    # Auto-start Ollama if using an Ollama model (primary or failover)
    ensure_ollama_running(args.model)
    if args.failover_model:
        ensure_ollama_running(args.failover_model)

    # Load .env file if present (API keys, secrets)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        logger.debug(f"Loading environment variables from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    # Set environment variables to handle corporate proxy SSL issues (Zscaler)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # Disable SSL verification for litellm/httpx (Zscaler proxy rewrites certs)
    os.environ.setdefault("LITELLM_SSL_VERIFY", "False")
    os.environ.setdefault("SSL_VERIFY", "False")

    # ── Merge database/scanner config ─────────────────────────────────────────
    if config:
        args.db_path = config.get("db_path", "./data/profiles.db")
        args.vector_store_path = config.get("vector_store_path", "./data/chroma")
        args.scanned_files_path = config.get("scanned_files_path", "./data/scanned_files")
        if args.scan_mode is None:
            args.scan_mode = config.get("scan_mode", "db_first")
    else:
        args.db_path = "./data/profiles.db"
        args.vector_store_path = "./data/chroma"
        args.scanned_files_path = "./data/scanned_files"
        if args.scan_mode is None:
            args.scan_mode = "db_first"

    # ── Route to the correct execution mode ───────────────────────────────────
    if args.db_status:
        # Show database status and exit
        _show_db_status(args)
    elif args.ingest and not args.match:
        # Ingest-only mode: scan resumes into DB
        asyncio.run(_run_ingest(args))
    elif args.match:
        # Match mode: query DB for candidates, score against JD
        if args.ingest:
            # --ingest --match: ingest first, then match
            asyncio.run(_run_ingest_and_match(args))
        else:
            asyncio.run(_run_match_from_db(args))
    else:
        # Default: original behavior (based on scan_mode)
        if args.scan_mode == "folder_only":
            # Bypass DB — run the original stateless pipeline
            asyncio.run(run_matching(args))
        elif args.scan_mode == "db_only":
            # Only match from DB (requires prior ingest)
            asyncio.run(_run_match_from_db(args))
        else:
            # "db_first" (default): ingest new files + match from DB
            asyncio.run(_run_ingest_and_match(args))


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TENANT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────


def _validate_tenant_flags(args, mode: str = "ingest") -> None:
    """
    Validate that --client-id and --job-id are provided for DB operations.
    These flags are mandatory for any mode that touches the DB (NDA enforcement).
    """
    missing = []
    if not args.client_id:
        missing.append("--client-id")
    if not args.job_id:
        missing.append("--job-id")

    if missing:
        print(f"\n  ERROR: {', '.join(missing)} required for --{mode} mode.")
        print(f"  These flags enforce NDA-level client isolation.")
        print(f"\n  Example:")
        print(f"    python run.py --{mode} --client-id ACME_CORP --job-id JOB-2024-001")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# NEW EXECUTION MODES
# ─────────────────────────────────────────────────────────────────────────────


def _show_db_status(args) -> None:
    """Display database and vector store status, then exit."""
    db = ProfileDatabase()
    vs = VectorStore()

    db_status = db.get_status()
    vs_status = vs.get_status()

    print("\n" + "=" * 50)
    print("DATABASE STATUS")
    print("=" * 50)
    print(f"  SQLite DB:       {db_status['db_path']}")
    print(f"  Profiles stored: {db_status['profile_count']}")
    print(f"  DB size:         {db_status['db_size_mb']} MB")
    print(f"  Scanned files:   {db_status['scanned_files_count']}")
    print(f"\n  Vector Store:    {vs_status['persist_path']}")
    print(f"  Embeddings:      {vs_status['embedding_count']}")
    print(f"  Collection:      {vs_status['collection_name']}")
    print("=" * 50)

    db.close()


async def _run_ingest(args) -> None:
    """Ingest resumes into the database (--ingest mode)."""
    print("\n" + "=" * 50)
    print("INGEST MODE — Scanning resumes into database")
    print(f"  Client ID: {args.client_id}")
    print(f"  Job ID:    {args.job_id}")
    print("=" * 50)

    db = ProfileDatabase()
    vs = VectorStore()

    result = await scan_and_ingest(
        resumes_dir=args.resumes,
        db=db,
        vector_store=vs,
        client_id=args.client_id,
        job_id=args.job_id,
        model=args.model,
        temperature=0.1,
    )

    print(f"\n  Ingest complete:")
    print(f"    New profiles:    {result['new_count']}")
    print(f"    Skipped (in DB): {result['skipped_count']}")
    print(f"    Failed:          {result['failed_count']}")
    print(f"    Total in DB:     {result['total_in_db']}")
    print(f"    Time:            {result['elapsed_seconds']}s")

    db.close()


async def _run_match_from_db(args) -> None:
    """
    Match mode: Query stored profiles from DB against a JD (--match mode).

    Fast — no LLM extraction needed for resumes (already in DB).
    Only JD extraction uses LLM (one call).
    """
    from matching_engine.jd_understanding import JDUnderstanding
    from matching_engine.scoring import Scorer
    from matching_engine.explainability import ExplainabilityEngine
    from matching_engine.semantic_matching import SemanticMatcher
    from matching_engine.models import MatchResult, ExplainabilityReport

    print("\n" + "=" * 50)
    print("MATCH MODE — Scoring stored profiles against JD")
    print("=" * 50)

    # ── Load JD ───────────────────────────────────────────────────────────────
    jd_text = load_jd(args)
    print(f"  JD loaded: {len(jd_text)} characters")

    # ── Load profiles from DB ─────────────────────────────────────────────────
    db = ProfileDatabase()
    vs = VectorStore()

    profile_count = db.get_profile_count()
    if profile_count == 0:
        print("\n  ERROR: No profiles in database. Run --ingest first.")
        print("    python run.py --ingest")
        db.close()
        return

    print(f"  Profiles in DB: {profile_count}")

    # ── Stage 1: Extract JD requirements (one LLM call) ──────────────────────
    print(f"  Extracting JD requirements...")
    jd_understanding = JDUnderstanding(model=args.model, temperature=0.1)
    jd = await jd_understanding.extract(jd_text)
    print(f"  JD parsed: {jd.title} ({len(jd.must_have_skills)} must-have skills)")

    # ── Query vector store for top candidates (semantic pre-filter) ───────────
    top_n_query = min(50, profile_count)  # Pre-filter top 50 by embedding similarity
    print(f"  Querying vector store for top {top_n_query} similar profiles...")

    similar_results = vs.query_similar(jd_text, client_id=args.client_id, top_n=top_n_query)

    if similar_results:
        # Get the file hashes of top candidates
        top_hashes = {r["file_hash"] for r in similar_results}
        # Load full profiles for these candidates
        all_profiles_meta = db.get_all_profiles_with_metadata(args.client_id)
        candidates = [p for p in all_profiles_meta if p["file_hash"] in top_hashes]
    else:
        # Fallback: use all profiles from DB
        candidates = db.get_all_profiles_with_metadata(args.client_id)

    print(f"  Candidates to score: {len(candidates)}")

    # ── Stages 3-4: Score each candidate (fast, no LLM needed) ────────────────
    print(f"  Scoring candidates...")
    scorer = Scorer()
    semantic_matcher = SemanticMatcher(embedding_model=args.embedding_model)

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

    # ── Sort by score ─────────────────────────────────────────────────────────
    results.sort(key=lambda r: r.qualification_percentage, reverse=True)

    # ── Stage 5: Explain top N ────────────────────────────────────────────────
    explain_count = args.explain_top or min(5, len(results))
    print(f"  Generating explanations for top {explain_count}...")
    explainability = ExplainabilityEngine(model=args.model, temperature=0.3)

    for result in results[:explain_count]:
        explanation = await explainability.explain(
            jd, result.candidate, result.scoring_breakdown, result.semantic_scores
        )
        result.explainability = explanation
        result.key_strengths = explanation.matched_strengths
        result.missing_skills = explanation.missing_skills
        result.reasoning = explanation.reason_for_score
        result.recommendation = explanation.recommendation

    # Fallback for remaining
    for result in results[explain_count:]:
        explanation = explainability._fallback_explanation(
            jd, result.candidate, result.scoring_breakdown
        )
        result.explainability = explanation
        result.key_strengths = explanation.matched_strengths
        result.missing_skills = explanation.missing_skills
        result.reasoning = explanation.reason_for_score
        result.recommendation = explanation.recommendation

    # ── Display results (reuse existing display logic) ────────────────────────
    # Build text_to_file mapping from DB metadata
    text_to_file = {
        cand["profile"].raw_text: {"filename": cand["source_file"], "path": cand["source_path"]}
        for cand in candidates
    }

    # Apply top-N filter
    if args.top:
        results = results[:args.top]

    _display_results(results, text_to_file)

    # Save JSON if requested
    if args.output:
        _save_json_output(results, text_to_file, args)

    db.close()
    print(f"\n  ✓ Match complete ({len(results)} candidates scored from DB)")


async def _run_ingest_and_match(args) -> None:
    """
    Combined mode: Ingest new files, then match all from DB (--ingest --match or default db_first).
    """
    # First ingest any new files
    db = ProfileDatabase()
    vs = VectorStore()

    print("\n  Checking for new resumes to ingest...")
    result = await scan_and_ingest(
        resumes_dir=args.resumes,
        db=db,
        vector_store=vs,
        model=args.model,
        temperature=0.1,
    )

    if result["new_count"] > 0:
        print(f"  Ingested {result['new_count']} new resumes. Total in DB: {result['total_in_db']}")
    else:
        print(f"  No new resumes. DB has {result['total_in_db']} profiles.")

    db.close()

    # Then run match from DB
    await _run_match_from_db(args)


def _display_results(results: list, text_to_file: dict) -> None:
    """Display the results table (extracted from run_matching for reuse)."""
    print("\n" + "=" * 180)
    print("RESULTS — Candidate Match Grid (sorted by % Qualified)")
    print("=" * 180)

    header = (
        f"{'#':<3} "
        f"{'Source File':<30} "
        f"{'First Name':<12} "
        f"{'Middle':<8} "
        f"{'Last Name':<15} "
        f"{'Contact Number':<18} "
        f"{'Email':<28} "
        f"{'Exp(Yrs)':<9} "
        f"{'% Match':<8} "
        f"{'Key Skills (Top 3)':<35} "
        f"{'Action'}"
    )
    print(header)
    print("-" * 180)

    for i, result in enumerate(results, 1):
        c = result.candidate
        file_info = text_to_file.get(c.raw_text, {"filename": f"unknown_{i}", "path": ""})
        source_file = file_info["filename"]
        if len(source_file) > 28:
            source_file = source_file[:25] + "..."

        first_name = c.first_name or "-"
        middle_name = c.middle_name or "-"
        last_name = c.last_name or "-"
        phone = c.phone or "N/A"
        email = c.email or "N/A"
        exp = f"{c.total_experience_years}" if c.total_experience_years else "N/A"
        score = f"{result.qualification_percentage}%"

        top_skills = ", ".join(c.skills[:3]) if c.skills else "N/A"
        if len(top_skills) > 33:
            top_skills = top_skills[:30] + "..."

        action = result.recommendation or "N/A"
        action = (
            action.replace("Strong Fit - ", "✅ ")
            .replace("Good Fit - ", "👍 ")
            .replace("Partial Fit - ", "⚠️  ")
            .replace("Weak Fit - ", "❌ ")
        )

        row = (
            f"{i:<3} "
            f"{source_file:<30} "
            f"{first_name:<12} "
            f"{middle_name:<8} "
            f"{last_name:<15} "
            f"{phone:<18} "
            f"{email:<28} "
            f"{exp:<9} "
            f"{score:<8} "
            f"{top_skills:<35} "
            f"{action}"
        )
        print(row)

    print("=" * 180)


def _save_json_output(results: list, text_to_file: dict, args) -> None:
    """Save results to JSON file."""
    import json

    output_data = {
        "metadata": {
            "jd_file": str(Path(args.jd).name) if args.jd else "auto-detected",
            "total_candidates": len(results),
            "model_used": args.model,
            "scan_mode": args.scan_mode,
        },
        "candidates": [],
    }

    for rank, r in enumerate(results, 1):
        file_info = text_to_file.get(r.candidate.raw_text, {"filename": "unknown", "path": ""})
        output_data["candidates"].append({
            "rank": rank,
            "source_file": file_info["filename"],
            "source_path": file_info["path"],
            "first_name": r.candidate.first_name or None,
            "middle_name": r.candidate.middle_name or None,
            "last_name": r.candidate.last_name or None,
            "full_name": r.candidate.full_name or None,
            "contact_number": r.candidate.phone or None,
            "email": r.candidate.email or None,
            "total_experience_years": r.candidate.total_experience_years,
            "qualification_percentage": r.qualification_percentage,
            "action": r.recommendation,
            "reasoning": r.reasoning,
            "key_skills_top_5": r.candidate.skills[:5],
            "matched_strengths": r.key_strengths,
            "missing_skills": r.missing_skills,
        })

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
