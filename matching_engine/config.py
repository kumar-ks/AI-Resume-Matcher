"""
Application Configuration for the AI Resume Matcher
=====================================================

Provides a Pydantic-based configuration model that replaces manual YAML loading.
Supports loading from YAML files, CLI argument merging, and validation.

Usage:
    from matching_engine.config import AppConfig

    # Load from YAML only
    config = AppConfig.from_yaml(Path("config.yaml"))

    # Load from YAML + CLI args (CLI takes priority)
    config = AppConfig.from_args_and_yaml(args, Path("config.yaml"))

    # Check model type
    if config.is_local_model:
        print("Using local Ollama model")

Called by:
    - run.py (main entry point)
    - matching_engine.pipeline (pipeline orchestration)
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class ScoringWeights(BaseModel):
    """
    Weights for each scoring dimension in the matching pipeline.

    These must sum to 1.0 and control how much each dimension contributes
    to the final qualification percentage.
    """

    must_have_match: float = 0.35
    experience_match: float = 0.25
    skills_depth: float = 0.20
    project_relevance: float = 0.12
    recency_factor: float = 0.08


class AppConfig(BaseModel):
    """
    Central application configuration.

    Consolidates all settings from config.yaml and CLI arguments into a single
    validated model. Provides helper properties for model type detection and
    classmethods for flexible loading.
    """

    # ── LLM Configuration ─────────────────────────────────────────────────────
    model: str = "ollama/llama3"
    failover_model: Optional[str] = "ollama/llama3"
    embedding_model: str = "all-MiniLM-L6-v2"
    temperature: float = 0.1
    max_tokens: int = 4096
    llm_timeout: int = 120  # seconds

    # ── Paths ─────────────────────────────────────────────────────────────────
    resumes_dir: str = "./resumes"
    jd_dir: str = "./jd"

    # ── Performance ───────────────────────────────────────────────────────────
    concurrency: int = 3
    explain_top: Optional[int] = None

    # ── Scoring ───────────────────────────────────────────────────────────────
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)

    # ── Output ────────────────────────────────────────────────────────────────
    output_file: Optional[str] = None
    top_n: Optional[int] = None
    debug: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_scoring_weights_sum(self) -> "AppConfig":
        """Validate that scoring weights sum to 1.0 (within floating-point tolerance)."""
        weights = self.scoring_weights
        total = (
            weights.must_have_match
            + weights.experience_match
            + weights.skills_depth
            + weights.project_relevance
            + weights.recency_factor
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"scoring_weights must sum to 1.0, got {total:.4f}. "
                f"Current values: must_have_match={weights.must_have_match}, "
                f"experience_match={weights.experience_match}, "
                f"skills_depth={weights.skills_depth}, "
                f"project_relevance={weights.project_relevance}, "
                f"recency_factor={weights.recency_factor}"
            )
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # PROPERTIES
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_local_model(self) -> bool:
        """Return True if the primary model is a local Ollama model."""
        return self.model.startswith("ollama/")

    @property
    def is_paid_model(self) -> bool:
        """
        Return True if the primary model is a cloud/paid provider.

        Detects OpenAI (gpt-*), Anthropic (anthropic/*), AWS Bedrock (bedrock/*),
        Azure (azure/*), and Google (gemini/*, vertex_ai/*) models.
        """
        paid_prefixes = (
            "gpt-",
            "anthropic/",
            "bedrock/",
            "azure/",
            "gemini/",
            "vertex_ai/",
            "openai/",
        )
        return any(self.model.startswith(prefix) for prefix in paid_prefixes)

    # ─────────────────────────────────────────────────────────────────────────
    # CLASSMETHODS — Loading
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            AppConfig instance with values from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
            ValidationError: If values fail Pydantic validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.debug("Loading configuration from %s", path)

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Handle nested scoring_weights dict from YAML
        config_data = dict(raw)
        if "scoring_weights" in config_data and isinstance(config_data["scoring_weights"], dict):
            config_data["scoring_weights"] = ScoringWeights(**config_data["scoring_weights"])

        logger.debug("Loaded config keys from YAML: %s", list(config_data.keys()))
        return cls(**config_data)

    @classmethod
    def from_args_and_yaml(cls, args, yaml_path: Path) -> "AppConfig":
        """
        Merge CLI arguments with YAML configuration. CLI takes priority.

        Loads the YAML file first as the base configuration, then overlays
        any non-default CLI argument values on top.

        Args:
            args: Parsed CLI arguments (argparse.Namespace or similar object
                  with attribute access). Only attributes that are not None
                  and differ from defaults override the YAML values.
            yaml_path: Path to the YAML configuration file.

        Returns:
            AppConfig instance with merged values (CLI > YAML > defaults).
        """
        # Start with YAML config (or defaults if YAML doesn't exist)
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            logger.debug("Loading base config from YAML: %s", yaml_path)
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        else:
            logger.debug("YAML config not found at %s, using defaults", yaml_path)
            yaml_data = {}

        # Convert args to dict (handles argparse.Namespace and similar objects)
        if hasattr(args, "__dict__"):
            args_dict = vars(args)
        elif hasattr(args, "__iter__"):
            args_dict = dict(args)
        else:
            args_dict = {}

        # Overlay CLI args onto YAML data — only non-None values override
        for key, value in args_dict.items():
            if value is not None:
                yaml_data[key] = value
                logger.debug("CLI override: %s = %r", key, value)

        # Handle nested scoring_weights
        if "scoring_weights" in yaml_data and isinstance(yaml_data["scoring_weights"], dict):
            yaml_data["scoring_weights"] = ScoringWeights(**yaml_data["scoring_weights"])

        return cls(**yaml_data)
