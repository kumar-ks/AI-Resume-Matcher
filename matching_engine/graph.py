"""
LangGraph Pipeline — Stateful Matching Workflow
=================================================

Refactors the linear MatchingPipeline into a LangGraph StateGraph with:
  - Typed state (MatchingState) flowing through nodes
  - Conditional routing (skip explanations for low scorers)
  - Checkpointed state for crash recovery (resume from last completed node)
  - Async execution with progress tracking

GRAPH STRUCTURE:
    START → parse_jd → extract_resumes → score_resumes → route_explain
                                                            │
                                         ┌──────────────────┼──────────────────┐
                                         ▼                                     ▼
                                   explain_results                     skip_explain
                                         │                                     │
                                         └──────────────────┬──────────────────┘
                                                            ▼
                                                     assemble_output → END

USAGE (API/CLI):
    from matching_engine.graph import run_matching_graph

    results = await run_matching_graph(
        jd_text="...",
        resume_texts=["...", "..."],
        client_id="ACME",
        job_id="JOB-001",
        model="ollama/llama3",
    )

CHECKPOINTING:
    By default uses MemorySaver (in-process). For production crash recovery,
    set USE_POSTGRES_CHECKPOINTER=true to use PostgreSQL-backed checkpointing
    via the existing DATABASE_URL connection.
"""

import asyncio
import logging
import time
import uuid
from typing import Annotated, Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from matching_engine.models import (
    ExplainabilityReport,
    JobDescription,
    MatchResult,
    ResumeProfile,
    ScoringBreakdown,
    SemanticMatchResult,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────


class MatchingState(TypedDict, total=False):
    """
    State flowing through the matching graph.

    Each node reads from and writes to this state dict.
    LangGraph handles state persistence and recovery.
    """
    # ── Inputs ──
    jd_text: str
    resume_texts: list[str]
    client_id: str
    job_id: str
    model: str
    temperature: float
    concurrency: int
    explain_top_n: Optional[int]

    # ── Stage 1 output ──
    jd: Optional[JobDescription]

    # ── Stage 2 output ──
    profiles: list[ResumeProfile]

    # ── Stage 3 output ──
    semantic_results: list[SemanticMatchResult]

    # ── Stage 4 output ──
    scoring_results: list[ScoringBreakdown]

    # ── Sorted intermediate ──
    sorted_indices: list[int]  # indices into profiles/scores sorted by score desc

    # ── Stage 5 output ──
    explanations: list[ExplainabilityReport]

    # ── Final output ──
    match_results: list[MatchResult]

    # ── Metadata ──
    start_time: float
    elapsed_seconds: float
    errors: list[str]
    current_stage: str


# ─────────────────────────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


async def parse_jd(state: MatchingState) -> dict:
    """
    Node: Stage 1 — Parse job description text into structured JobDescription.
    """
    from matching_engine.jd_understanding import JDUnderstanding

    logger.info("Graph node: parse_jd")
    model = state.get("model", "ollama/llama3")
    temperature = state.get("temperature", 0.1)

    engine = JDUnderstanding(model=model, temperature=temperature)
    jd = await engine.extract(state["jd_text"])

    logger.info(f"JD parsed: {jd.title} ({len(jd.must_have_skills)} must-have skills)")
    return {"jd": jd, "current_stage": "jd_parsed"}


async def extract_resumes(state: MatchingState) -> dict:
    """
    Node: Stage 2 — Extract structured profiles from all resume texts concurrently.
    """
    from matching_engine.resume_understanding import ResumeUnderstanding

    logger.info("Graph node: extract_resumes")
    model = state.get("model", "ollama/llama3")
    temperature = state.get("temperature", 0.1)
    concurrency = state.get("concurrency", 3)
    resume_texts = state.get("resume_texts", [])

    engine = ResumeUnderstanding(model=model, temperature=temperature)
    semaphore = asyncio.Semaphore(concurrency)

    async def extract_one(text: str) -> ResumeProfile:
        async with semaphore:
            return await engine.extract(text)

    tasks = [extract_one(text) for text in resume_texts]
    profiles = await asyncio.gather(*tasks)

    logger.info(f"Extracted {len(profiles)} resume profiles")
    return {"profiles": list(profiles), "current_stage": "resumes_extracted"}


async def score_resumes(state: MatchingState) -> dict:
    """
    Node: Stages 3+4 — Semantic matching + scoring for all profiles.

    Runs synchronously (CPU-bound embedding computation, no async benefit).
    Produces sorted indices for downstream nodes.
    """
    from matching_engine.scoring import Scorer
    from matching_engine.semantic_matching import SemanticMatcher

    logger.info("Graph node: score_resumes")
    jd = state["jd"]
    profiles = state.get("profiles", [])

    matcher = SemanticMatcher()
    scorer = Scorer()

    semantic_results = []
    scoring_results = []

    for profile in profiles:
        sem = matcher.match(jd, profile)
        score = scorer.score(jd, profile, sem)
        semantic_results.append(sem)
        scoring_results.append(score)

    # Sort by qualification_percentage descending
    indexed_scores = list(enumerate(scoring_results))
    indexed_scores.sort(key=lambda x: x[1].qualification_percentage, reverse=True)
    sorted_indices = [idx for idx, _ in indexed_scores]

    if scoring_results:
        top_idx = sorted_indices[0]
        logger.info(
            f"Scoring complete. Top: {profiles[top_idx].full_name} "
            f"({scoring_results[top_idx].qualification_percentage}%)"
        )

    return {
        "semantic_results": semantic_results,
        "scoring_results": scoring_results,
        "sorted_indices": sorted_indices,
        "current_stage": "scored",
    }


def route_explain(state: MatchingState) -> str:
    """
    Conditional edge: decide whether to run LLM explanations or skip.

    Skips explanations if explain_top_n is explicitly set to 0.
    """
    explain_top_n = state.get("explain_top_n")
    if explain_top_n == 0:
        logger.info("Explanations skipped (explain_top_n=0)")
        return "skip_explain"
    return "explain_results"


async def explain_results(state: MatchingState) -> dict:
    """
    Node: Stage 5 — Generate LLM explanations for top N candidates.

    Only the top N (by score) get LLM-generated explanations.
    The rest get rule-based fallback explanations.
    """
    from matching_engine.explainability import ExplainabilityEngine

    logger.info("Graph node: explain_results")
    model = state.get("model", "ollama/llama3")
    jd = state["jd"]
    profiles = state.get("profiles", [])
    scoring_results = state.get("scoring_results", [])
    semantic_results = state.get("semantic_results", [])
    sorted_indices = state.get("sorted_indices", [])
    explain_top_n = state.get("explain_top_n") or len(profiles)

    engine = ExplainabilityEngine(model=model, temperature=0.3)

    # Pre-fill explanations list (one per profile, in original order)
    explanations: list[ExplainabilityReport] = [ExplainabilityReport()] * len(profiles)

    # LLM explanations for top N (by sorted rank)
    for rank, idx in enumerate(sorted_indices[:explain_top_n]):
        explanation = await engine.explain(
            jd, profiles[idx], scoring_results[idx], semantic_results[idx]
        )
        explanations[idx] = explanation

    # Rule-based fallback for the rest
    for idx in sorted_indices[explain_top_n:]:
        explanation = engine._fallback_explanation(jd, profiles[idx], scoring_results[idx])
        explanations[idx] = explanation

    logger.info(f"Explanations generated: {explain_top_n} LLM + {max(0, len(profiles) - explain_top_n)} fallback")
    return {"explanations": explanations, "current_stage": "explained"}


async def skip_explain(state: MatchingState) -> dict:
    """
    Node: Skip explanations — use rule-based fallback for all candidates.
    """
    from matching_engine.explainability import ExplainabilityEngine

    logger.info("Graph node: skip_explain (all rule-based)")
    model = state.get("model", "ollama/llama3")
    jd = state["jd"]
    profiles = state.get("profiles", [])
    scoring_results = state.get("scoring_results", [])

    engine = ExplainabilityEngine(model=model, temperature=0.3)
    explanations = [
        engine._fallback_explanation(jd, profiles[idx], scoring_results[idx])
        for idx in range(len(profiles))
    ]

    return {"explanations": explanations, "current_stage": "explained"}


async def assemble_output(state: MatchingState) -> dict:
    """
    Node: Stage 6 — Assemble final MatchResult list sorted by score.
    """
    logger.info("Graph node: assemble_output")
    jd = state["jd"]
    profiles = state.get("profiles", [])
    semantic_results = state.get("semantic_results", [])
    scoring_results = state.get("scoring_results", [])
    explanations = state.get("explanations", [])
    sorted_indices = state.get("sorted_indices", [])
    start_time = state.get("start_time", time.time())

    results: list[MatchResult] = []
    for idx in sorted_indices:
        explanation = explanations[idx] if idx < len(explanations) else ExplainabilityReport()
        result = MatchResult(
            candidate=profiles[idx],
            job_description=jd,
            qualification_percentage=scoring_results[idx].qualification_percentage,
            semantic_scores=semantic_results[idx],
            scoring_breakdown=scoring_results[idx],
            explainability=explanation,
            key_strengths=explanation.matched_strengths,
            missing_skills=explanation.missing_skills,
            reasoning=explanation.reason_for_score,
            recommendation=explanation.recommendation,
        )
        results.append(result)

    elapsed = time.time() - start_time
    logger.info(f"Graph complete in {elapsed:.1f}s. {len(results)} results assembled.")

    return {
        "match_results": results,
        "elapsed_seconds": elapsed,
        "current_stage": "complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────


def build_matching_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for the matching pipeline.

    Returns an uncompiled graph. Call .compile(checkpointer=...) to get a runnable.
    """
    graph = StateGraph(MatchingState)

    # Add nodes
    graph.add_node("parse_jd", parse_jd)
    graph.add_node("extract_resumes", extract_resumes)
    graph.add_node("score_resumes", score_resumes)
    graph.add_node("explain_results", explain_results)
    graph.add_node("skip_explain", skip_explain)
    graph.add_node("assemble_output", assemble_output)

    # Add edges
    graph.add_edge(START, "parse_jd")
    graph.add_edge("parse_jd", "extract_resumes")
    graph.add_edge("extract_resumes", "score_resumes")

    # Conditional: after scoring, decide whether to explain
    graph.add_conditional_edges(
        "score_resumes",
        route_explain,
        {
            "explain_results": "explain_results",
            "skip_explain": "skip_explain",
        },
    )

    # Both explain paths lead to assemble_output
    graph.add_edge("explain_results", "assemble_output")
    graph.add_edge("skip_explain", "assemble_output")

    graph.add_edge("assemble_output", END)

    return graph


def compile_graph(checkpointer=None):
    """
    Compile the matching graph with an optional checkpointer.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Use MemorySaver() for dev or PostgresSaver for production.

    Returns:
        A compiled, runnable graph.
    """
    graph = build_matching_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Matching graph compiled")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH RUNNER — the public interface
# ─────────────────────────────────────────────────────────────────────────────

# Module-level compiled graph (lazy initialization)
_compiled_graph = None


def _get_graph():
    """Get or create the module-level compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_graph()
    return _compiled_graph


async def run_matching_graph(
    jd_text: str,
    resume_texts: list[str],
    client_id: str = "",
    job_id: str = "",
    model: str = "ollama/llama3",
    temperature: float = 0.1,
    concurrency: int = 3,
    explain_top_n: Optional[int] = None,
    thread_id: Optional[str] = None,
) -> list[MatchResult]:
    """
    Run the matching pipeline as a LangGraph workflow.

    This is the primary public API for the graph-based pipeline.
    It replaces MatchingPipeline.match() for new integrations.

    Args:
        jd_text: Raw job description text
        resume_texts: List of raw resume texts
        client_id: Client identifier (for tracing/logging)
        job_id: Job identifier (for tracing/logging)
        model: LiteLLM model identifier
        temperature: LLM temperature
        concurrency: Max parallel resume extractions
        explain_top_n: Only LLM-explain top N candidates (None = all)
        thread_id: Optional thread ID for checkpointed resumption

    Returns:
        List of MatchResult sorted by qualification_percentage (descending)
    """
    graph = _get_graph()

    # Initial state
    initial_state: MatchingState = {
        "jd_text": jd_text,
        "resume_texts": resume_texts,
        "client_id": client_id,
        "job_id": job_id,
        "model": model,
        "temperature": temperature,
        "concurrency": concurrency,
        "explain_top_n": explain_top_n,
        "start_time": time.time(),
        "errors": [],
        "current_stage": "starting",
    }

    # Config for checkpointing (thread_id enables resumption)
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    logger.info(
        f"Running matching graph: {len(resume_texts)} resumes, "
        f"model={model}, client={client_id}, job={job_id}"
    )

    # Execute the graph
    final_state = await graph.ainvoke(initial_state, config=config)

    results = final_state.get("match_results", [])
    elapsed = final_state.get("elapsed_seconds", 0)

    if results:
        print(f"\n  ⏱  Graph pipeline completed in {elapsed:.1f} seconds ({len(resume_texts)} resumes)")

    return results
