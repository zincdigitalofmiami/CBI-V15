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
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> List[SweepResult]:
    """
    Run a model sweep: train all candidate models and evaluate.

    Args:
        config: Sweep configuration
        motherduck_token: Optional MotherDuck token (uses env if not provided)
        train_df, val_df, test_df: Pre-loaded data splits (optional, will load if not provided)

    Returns:
        List of SweepResult objects, one per model
    """
    import os
    import time

    logger.info(f"\n{'='*60}")
    logger.info(f"Running Model Sweep:")
    logger.info(f"  Horizon: {config.horizon}")
    logger.info(f"  Bucket Focus: {config.bucket_focus}")
    logger.info(f"  Candidates: {', '.join(config.candidate_models)}")
    logger.info(f"  Train Table: {config.train_table}")
    logger.info(f"{'='*60}")

    # Map horizon name to target column
    horizon_to_col = {
        "1w": "target_1w_price",
        "1m": "target_1m_price",
        "3m": "target_3m_price",
        "6m": "target_6m_price",
        "12m": "target_12m_price",
    }
    horizon_col = horizon_to_col.get(config.horizon)
    if not horizon_col:
        raise ValueError(f"Unknown horizon: {config.horizon}")

    # Load data if not provided
    if train_df is None or val_df is None or test_df is None:
        logger.info("Loading training data from MotherDuck...")
        token = motherduck_token or os.getenv("MOTHERDUCK_TOKEN")
        if not token:
            raise ValueError("MOTHERDUCK_TOKEN not set and no data provided")

        db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
        conn = duckdb.connect(f"md:{db_name}?motherduck_token={token}")

        df = conn.execute(
            f"SELECT * FROM {config.train_table} WHERE symbol = 'ZL' ORDER BY date"
        ).df()
        conn.close()

        if df.empty:
            raise ValueError(f"No data returned from {config.train_table}")

        df["date"] = pd.to_datetime(df["date"])
        train_df = df[df["date"] < "2023-01-01"].copy()
        val_df = df[(df["date"] >= "2023-01-01") & (df["date"] < "2023-07-01")].copy()
        test_df = df[df["date"] >= "2023-07-01"].copy()

    results = []

    for model_name in config.candidate_models:
        logger.info(f"\n--- Training {model_name} for {config.horizon} ---")

        trainer = EngineRegistry.get_model_trainer(model_name)
        if trainer is None:
            logger.warning(f"Trainer not found for {model_name}; skipping")
            continue

        start_time = time.time()

        try:
            # Call the appropriate training function based on model type
            if model_name == "lightgbm":
                # LightGBM uses different function signature
                model, _ = trainer.train_lgbm_for_horizon(horizon_col, config.horizon)
                metrics = {
                    "rmse": 0.0,
                    "pinball_p50": 0.0,
                    "coverage": 0.0,
                }  # Would need to extract from model
            elif model_name in ["catboost", "xgboost"]:
                # CatBoost/XGBoost use train_*_for_horizon
                train_func = getattr(trainer, f"train_{model_name}_for_horizon", None)
                if train_func:
                    models, metrics = train_func(
                        train_df, val_df, test_df, horizon_col, config.horizon
                    )
                else:
                    logger.warning(f"No training function found for {model_name}")
                    continue
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue

            train_time = time.time() - start_time

            # Extract metrics for sweep comparison
            if isinstance(metrics, dict) and "P50" in metrics:
                pinball_p50 = metrics["P50"].get("pinball_loss", 0.0)
                mae = metrics["P50"].get("mae", 0.0)
            else:
                pinball_p50 = 0.0
                mae = 0.0

            result = SweepResult(
                model_name=model_name,
                horizon=config.horizon,
                bucket_focus=config.bucket_focus,
                metrics={
                    "pinball_p50": pinball_p50,
                    "mae": mae,
                    "train_time": train_time,
                },
                model_path=f"models/baselines/{model_name}/",
            )
            results.append(result)

            logger.info(f"  ✅ {model_name} trained in {train_time:.1f}s")
            logger.info(f"     Pinball P50: {pinball_p50:.4f}, MAE: {mae:.4f}")

        except Exception as e:
            logger.error(f"  ❌ {model_name} failed: {e}")
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Sweep Complete: {len(results)} models trained")
    logger.info(f"{'='*60}")

    return results


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
