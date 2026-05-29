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

from matching_engine.file_loader import extract_text, load_files_from_directory
from matching_engine.pipeline import MatchingPipeline

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# Default: INFO level. Use --debug flag to enable DEBUG level for all modules.
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the entire application.

    Called by: main()

    Args:
        debug: If True, sets log level to DEBUG for verbose output.
               If False, sets log level to INFO (default).

    Log Levels:
        DEBUG  - Detailed internal state (LLM responses, regex matches, scores)
        INFO   - Pipeline progress (stage transitions, file loading)
        WARNING - Non-fatal issues (LLM retry, fallback used)
        ERROR  - Failures that affect output quality
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers unless in debug mode
    if not debug:
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("numexpr").setLevel(logging.WARNING)


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

Folder structure:
  AI-Resume-Matcher/
  ├── resumes/          ← Place candidate resumes here
  │   ├── rohit_sharma.pdf
  │   ├── anita_iyer.docx
  │   └── vikram_reddy.txt
  ├── jd/               ← Place job description here
  │   └── senior_backend_developer.pdf
  └── run.py
        """,
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
    # ── Step 1: Load input files ──────────────────────────────────────────────
    jd_text = load_jd(args)
    resume_files = load_resumes(args)

    # ── Step 2: Display configuration banner ──────────────────────────────────
    print("\n" + "=" * 70)
    print("AI RESUME MATCHER")
    print("=" * 70)
    print(f"  JD loaded: {len(jd_text)} characters")
    print(f"  Resumes found: {len(resume_files)}")
    print(f"  Model: {args.model}")
    print(f"  Embeddings: {args.embedding_model}")
    print(f"  Debug mode: {'ON' if args.debug else 'OFF'}")
    print("=" * 70)

    # ── Step 3: Initialize the matching pipeline ──────────────────────────────
    # This creates instances of all 5 stage engines:
    #   - JDUnderstanding (Stage 1)
    #   - ResumeUnderstanding (Stage 2)
    #   - SemanticMatcher (Stage 3) — also pre-loads the embedding model
    #   - Scorer (Stage 4)
    #   - ExplainabilityEngine (Stage 5)
    pipeline = MatchingPipeline(
        model=args.model,
        embedding_model=args.embedding_model,
    )

    # ── Step 4: Run the matching pipeline ─────────────────────────────────────
    # pipeline.match() executes stages 1-6 for each resume against the JD
    resume_texts = [f["text"] for f in resume_files]
    results = await pipeline.match(jd_text=jd_text, resume_texts=resume_texts)

    # ── Step 5: Build filename fallback mapping ───────────────────────────────
    # When LLM fails to extract a candidate name, we fall back to the filename.
    # We map raw_text → filename because results are sorted (original index lost).
    text_to_filename = {f["text"]: f["filename"] for f in resume_files}

    # ── Step 6: Apply top-N filter if requested ───────────────────────────────
    if args.top:
        results = results[: args.top]

    # ── Step 7: Display results table ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — Candidate Match Grid (sorted by Match %)")
    print("=" * 70)
    print(f"{'#':<4} {'Name':<25} {'Exp':<8} {'Match %':<10} {'Recommendation'}")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        # Use extracted name, fall back to filename if extraction failed
        name = result.candidate.full_name or text_to_filename.get(
            result.candidate.raw_text, f"Candidate {i}"
        )
        exp = (
            f"{result.candidate.total_experience_years}y"
            if result.candidate.total_experience_years
            else "N/A"
        )
        score = f"{result.qualification_percentage}%"
        rec = result.recommendation
        print(f"{i:<4} {name:<25} {exp:<8} {score:<10} {rec}")

    # ── Step 8: Display detailed view for top candidate ───────────────────────
    if results:
        top = results[0]
        print("\n" + "=" * 70)
        print(f"TOP CANDIDATE DETAIL")
        print("=" * 70)
        print(f"  Name:          {top.candidate.full_name}")
        print(f"  Email:         {top.candidate.email or 'N/A'}")
        print(f"  Phone:         {top.candidate.phone or 'N/A'}")
        print(f"  Experience:    {top.candidate.total_experience_years or 'N/A'} years")
        print(f"  Match Score:   {top.qualification_percentage}%")
        print(f"\n  AI Reasoning:")
        print(f"    {top.reasoning}")
        print(f"\n  Matched Strengths:")
        for s in top.key_strengths:
            print(f"    + {s}")
        print(f"\n  Missing / Gap Areas:")
        for s in top.missing_skills:
            print(f"    - {s}")
        print(f"\n  Recommendation: {top.recommendation}")

    # ── Step 9: Save to JSON if --output specified ────────────────────────────
    if args.output:
        import json

        output_data = []
        for r in results:
            output_data.append({
                "name": r.candidate.full_name,
                "email": r.candidate.email,
                "phone": r.candidate.phone,
                "experience_years": r.candidate.total_experience_years,
                "qualification_percentage": r.qualification_percentage,
                "key_strengths": r.key_strengths,
                "missing_skills": r.missing_skills,
                "reasoning": r.reasoning,
                "recommendation": r.recommendation,
                "scoring_breakdown": {
                    "must_have_match": r.scoring_breakdown.must_have_match,
                    "experience_match": r.scoring_breakdown.experience_match,
                    "skills_depth": r.scoring_breakdown.skills_depth,
                    "project_relevance": r.scoring_breakdown.project_relevance,
                    "recency_factor": r.scoring_breakdown.recency_factor,
                },
            })
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def main():
    """
    Application entry point.

    Flow: parse_args() → setup_logging() → ensure_ollama_running() → asyncio.run(run_matching())
    """
    args = parse_args()
    setup_logging(debug=args.debug)

    # Auto-start Ollama if using an Ollama model and server isn't running
    ensure_ollama_running(args.model)

    # Set environment variable to skip remote model cost map fetch (avoids SSL issues)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    asyncio.run(run_matching(args))


if __name__ == "__main__":
    main()
