#!/usr/bin/env python3
"""
Quantile Regression Averaging (QRA) Ensemble

Combines quantile forecasts from multiple models using weighted averaging.
Preserves the full uncertainty structure (P10, P50, P90) from component models.

Reference: Nowotarski & Weron (2018), "Recent advances in electricity price forecasting"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QuantileForecast:
    """A single model's quantile forecast."""

    model_name: str
    horizon: str
    dates: pd.Series
    p10: np.ndarray  # 10th percentile
    p50: np.ndarray  # 50th percentile (median)
    p90: np.ndarray  # 90th percentile

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame(
            {
                "date": self.dates,
                "p10": self.p10,
                "p50": self.p50,
                "p90": self.p90,
                "model": self.model_name,
            }
        )


@dataclass
class EnsembleForecast:
    """QRA ensemble result."""

    horizon: str
    dates: pd.Series
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    weights: Dict[str, float]  # Model weights used
    regime: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for storage."""
        return pd.DataFrame(
            {
                "date": self.dates,
                "forecast_p10": self.p10,
                "forecast_p50": self.p50,
                "forecast_p90": self.p90,
                "horizon": self.horizon,
                "regime": self.regime,
            }
        )


def run_qra(
    forecasts: List[QuantileForecast],
    weights: Dict[str, float],
    regime: str = "adaptive",
) -> EnsembleForecast:
    """
    Run Quantile Regression Averaging to combine forecasts.

    Args:
        forecasts: List of QuantileForecast objects from each model
        weights: Dict mapping model names to ensemble weights (must sum to ~1.0)
        regime: Current market regime (for logging/metadata)

    Returns:
        EnsembleForecast with combined quantiles
    """
    if not forecasts:
        raise ValueError("No forecasts provided for QRA")

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Weights sum to zero")

    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    logger.info(f"\n{'='*60}")
    logger.info("Running Quantile Regression Averaging (QRA)")
    logger.info(f"  Regime: {regime}")
    logger.info(f"  Models: {len(forecasts)}")
    logger.info("  Weights:")
    for model, weight in normalized_weights.items():
        logger.info(f"    {model}: {weight:.3f}")
    logger.info(f"{'='*60}")

    # Use the first forecast's dates as template
    base_dates = forecasts[0].dates
    n = len(base_dates)

    # Initialize weighted sums
    ensemble_p10 = np.zeros(n)
    ensemble_p50 = np.zeros(n)
    ensemble_p90 = np.zeros(n)

    # Weighted average of quantiles
    for forecast in forecasts:
        weight = normalized_weights.get(forecast.model_name, 0.0)
        if weight > 0:
            # Ensure dates align (in production, would need robust date matching)
            if len(forecast.p10) == n:
                ensemble_p10 += weight * forecast.p10
                ensemble_p50 += weight * forecast.p50
                ensemble_p90 += weight * forecast.p90
            else:
                logger.warning(
                    f"  ⚠️  {forecast.model_name} has {len(forecast.p10)} points, "
                    f"expected {n}; skipping"
                )

    result = EnsembleForecast(
        horizon=forecasts[0].horizon,
        dates=base_dates,
        p10=ensemble_p10,
        p50=ensemble_p50,
        p90=ensemble_p90,
        weights=normalized_weights,
        regime=regime,
    )

    logger.info("✅ QRA complete")
    logger.info(f"   Output: {len(result.dates)} forecast points")
    logger.info(f"   P50 range: [{result.p50.min():.2f}, {result.p50.max():.2f}]")

    return result


def calculate_interval_score(
    actual: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Calculate interval score for probabilistic forecast evaluation.

    Lower is better. Penalizes both width and coverage violations.

    Args:
        actual: Actual values
        p10: 10th percentile forecast
        p90: 90th percentile forecast
        alpha: Coverage level (0.1 for 90% interval)

    Returns:
        Interval score (lower is better)
    """
    width = p90 - p10
    lower_penalty = 2 * (1 / alpha) * np.maximum(0, p10 - actual)
    upper_penalty = 2 * (1 / alpha) * np.maximum(0, actual - p90)

    score = width + lower_penalty + upper_penalty
    return float(np.mean(score))
