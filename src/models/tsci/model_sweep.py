#!/usr/bin/env python3
"""
TSci Model Sweep Module

Lightweight AutoML-style sweep that:
- Trains multiple candidate models per bucket/horizon
- Evaluates on validation set
- Records results to tsci.runs
- Selects winner per bucket/horizon

Big 8 buckets are focus overlays for reporting and tagging;
models see the FULL feature set from Anofox.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from src.engines.engine_registry import EngineRegistry

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for a model sweep."""

    horizon: str  # e.g., "1w", "1m", "3m"
    bucket_focus: str  # e.g., "crush", "volatility" (for tagging, NOT filtering)
    candidate_models: List[str]  # e.g., ["lightgbm", "catboost", "xgboost"]
    train_table: str  # e.g., "training.daily_ml_matrix_zl"
    regime: str = "adaptive"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "horizon": self.horizon,
            "bucket_focus": self.bucket_focus,
            "candidate_models": self.candidate_models,
            "train_table": self.train_table,
            "regime": self.regime,
        }


@dataclass
class SweepResult:
    """Result from a single model in a sweep."""

    model_name: str
    horizon: str
    bucket_focus: str
    metrics: Dict[str, float]  # rmse, pinball, coverage, train_time
    model_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for tsci.runs."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "bucket_focus": self.bucket_focus,
            **self.metrics,
            "model_path": self.model_path,
        }


def run_model_sweep(
    config: SweepConfig,
    motherduck_token: Optional[str] = None,
) -> List[SweepResult]:
    """
    Run a model sweep: train all candidate models and evaluate.

    Args:
        config: Sweep configuration
        motherduck_token: Optional MotherDuck token (uses env if not provided)

    Returns:
        List of SweepResult objects, one per model
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Model Sweep:")
    logger.info(f"  Horizon: {config.horizon}")
    logger.info(f"  Bucket Focus: {config.bucket_focus}")
    logger.info(f"  Candidates: {', '.join(config.candidate_models)}")
    logger.info(f"  Train Table: {config.train_table}")
    logger.info(f"{'='*60}")

    # NOTE: This function previously returned mock metrics. To avoid polluting
    # tsci.runs with fake data, it now raises NotImplementedError until the
    # training pipeline is fully wired to real baselines.

    logger.info("TSci model sweep requested with config: %s", config.to_dict())
    logger.info("Training data table (MotherDuck): %s", config.train_table)
    logger.info("Candidate models: %s", ", ".join(config.candidate_models))

    # Validate that trainers are registered, so misconfigurations fail early.
    for model_name in config.candidate_models:
        trainer = EngineRegistry.get_model_trainer(model_name)
        if trainer is None:
            logger.warning(
                "Trainer not found for %s; update EngineRegistry.MODEL_FAMILIES",
                model_name,
            )

    raise NotImplementedError(
        "run_model_sweep currently does not perform training or return mock metrics. "
        "Wire this function to the real baselines (LightGBM/CatBoost/XGBoost) "
        "using training.daily_ml_matrix_zl before use."
    )


def select_best_model(
    results: List[SweepResult], metric: str = "pinball_p50"
) -> Optional[SweepResult]:
    """
    Select the best model from sweep results.

    Args:
        results: List of SweepResult objects
        metric: Metric to optimize (lower is better)

    Returns:
        Best SweepResult or None if results empty
    """
    if not results:
        return None

    # Find model with lowest metric
    best = min(results, key=lambda r: r.metrics.get(metric, float("inf")))
    logger.info(
        f"  🏆 Winner: {best.model_name} ({metric}={best.metrics.get(metric):.4f})"
    )

    return best


def log_results_to_motherduck(
    results: List[SweepResult],
    motherduck_token: Optional[str] = None,
) -> None:
    """
    Log sweep results to tsci.runs table in MotherDuck.

    Args:
        results: List of SweepResult objects
        motherduck_token: Optional MotherDuck token
    """
    if not results:
        logger.warning("No results to log")
        return

    # TODO: Implement actual MotherDuck logging
    logger.info(f"📝 Would log {len(results)} results to tsci.runs")
    for result in results:
        logger.info(f"   - {result.model_name}/{result.horizon}: {result.metrics}")
