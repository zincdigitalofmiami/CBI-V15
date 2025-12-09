"""
TSci: Planner Agent

Plans experiments, model sweeps, and orchestrates the training pipeline.
Writes to tsci.jobs for scheduling.

Uses OpenAI to suggest model candidates and hyperparameter bands based on
bucket characteristics, regime labels, and recent performance. The LLM
only suggests; TSci owns the actual job creation.
"""

from __future__ import annotations

import duckdb
import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.utils.openai_client import run_chat

logger = logging.getLogger(__name__)

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")


def create_job(
    con: duckdb.DuckDBPyConnection, job_name: str, job_type: str, config: dict = None
) -> int:
    """
    Create a new job in tsci.jobs.

    Returns:
        job_id of the created job
    """
    result = con.execute(
        "SELECT COALESCE(MAX(job_id), 0) + 1 FROM tsci.jobs"
    ).fetchone()
    job_id = result[0]

    con.execute(
        """
        INSERT INTO tsci.jobs (job_id, job_name, job_type, config_json)
        VALUES (?, ?, ?, ?)
    """,
        [job_id, job_name, job_type, str(config) if config else None],
    )

    return job_id


def suggest_model_candidates(
    bucket: str,
    horizon: str,
    regime: str,
    recent_metrics: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Suggest model candidates and hyperparameter bands for a given bucket/horizon.

    Args:
        bucket: Bucket name (e.g., 'crush', 'china', 'volatility')
        horizon: Forecast horizon (e.g., '1w', '1m', '3m')
        regime: Current market regime
        recent_metrics: Optional dict of recent model performance
        use_llm: Whether to use OpenAI for suggestions (default True)

    Returns:
        Dict with candidate_models, hyperparam_ranges, focus_features
    """
    if not use_llm:
        # Fallback: simple heuristic based on bucket
        if bucket in ["volatility", "tariff"]:
            return {
                "candidate_models": ["lightgbm", "catboost", "tft"],
                "hyperparam_ranges": {"depth": [3, 7], "lr": [0.01, 0.1]},
                "focus_features": ["all"],  # Big 8 as focus, not filter
            }
        return {
            "candidate_models": ["lightgbm", "catboost"],
            "hyperparam_ranges": {"depth": [3, 7], "lr": [0.01, 0.1]},
            "focus_features": ["all"],
        }

    # LLM-assisted suggestion
    system_prompt = (
        "You are a model-selection strategist for commodity futures forecasting. "
        "Given a bucket (driver category), horizon, and regime, recommend:\n"
        "1) candidate_models: array of model names (lightgbm, catboost, xgboost, tft, prophet)\n"
        "2) hyperparam_ranges: compact ranges for depth, learning_rate, etc.\n"
        "3) focus_features: 'all' (models see all features; Big 8 are tags, not filters)\n"
        "RETURN JSON ONLY. NEVER invent table names or metrics not provided.\n"
        "The Big 8 buckets (Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility) "
        "are emphasis overlays; models are NOT restricted to seeing only those features."
    )

    payload = {
        "bucket": bucket,
        "horizon": horizon,
        "regime": regime,
        "recent_metrics": recent_metrics or {},
    }

    try:
        response_text = run_chat(
            prompt=json.dumps(payload),
            system=system_prompt,
            temperature=0.2,
        )
        result = json.loads(response_text)
        if not isinstance(result, dict):
            raise ValueError("LLM response was not JSON")
        return result
    except Exception as exc:
        logger.warning("Planner LLM call failed, using fallback: %s", exc)
        return {
            "candidate_models": ["lightgbm", "catboost"],
            "hyperparam_ranges": {"depth": [3, 7], "lr": [0.01, 0.1]},
            "focus_features": ["all"],
        }


def plan_training_sweep(
    horizons: Optional[List[str]] = None,
    buckets: Optional[List[str]] = None,
    use_llm: bool = True,
) -> None:
    """
    Plan a training sweep across horizons and buckets.

    Args:
        horizons: List of horizons (default: 1w, 1m, 3m, 6m, 12m)
        buckets: Optional Big 8 focus buckets (default: all 8)
        use_llm: Whether to use OpenAI for model suggestions
    """
    if horizons is None:
        horizons = ["1w", "1m", "3m", "6m", "12m"]

    if buckets is None:
        # Big 8 as default focus, but models see all features
        buckets = [
            "crush",
            "china",
            "fx",
            "fed",
            "tariff",
            "biofuel",
            "energy",
            "volatility",  # ALWAYS "volatility", never "vol"
        ]

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}")

    for horizon in horizons:
        for bucket in buckets:
            # Get model suggestions (optionally from LLM)
            suggestions = suggest_model_candidates(
                bucket=bucket,
                horizon=horizon,
                regime="adaptive",  # TODO: fetch from MotherDuck
                use_llm=use_llm,
            )

            job_config = {
                "horizon": horizon,
                "symbol": "ZL",
                "bucket_focus": bucket,
                "candidate_models": suggestions.get("candidate_models", ["lightgbm"]),
                "hyperparam_ranges": suggestions.get("hyperparam_ranges", {}),
                "focus_features": "all",  # Models see all features
            }

            job_id = create_job(
                con,
                job_name=f"Sweep - {bucket}/{horizon}",
                job_type="training",
                config=job_config,
            )
            logger.info(f"Created job {job_id}: {bucket}/{horizon}")

    con.close()


if __name__ == "__main__":
    plan_training_sweep()
