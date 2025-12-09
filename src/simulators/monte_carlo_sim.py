#!/usr/bin/env python3
"""
Monte Carlo Risk Simulator

Generates probabilistic price paths from forecast quantiles (P10, P50, P90).
Stress-tests scenarios, estimates downside risk, simulates path outcomes.

This is L4 in the modeling stack: takes the QRA ensemble output and
produces scenario distributions for risk analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""

    n_paths: int = 1000  # Number of simulation paths
    random_seed: int = 42
    horizons: List[str] = None  # e.g., ["1w", "1m", "3m"]

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = ["1w", "1m", "3m", "6m", "12m"]


@dataclass
class SimulationResult:
    """Monte Carlo simulation result."""

    horizon: str
    paths: np.ndarray  # Shape: (n_paths, n_periods)
    percentiles: Dict[
        int, np.ndarray
    ]  # e.g., {5: [...], 10: [...], 50: [...], 90: [...], 95: [...]}
    downside_risk: Dict[str, float]  # VaR, CVaR, max_drawdown
    scenario_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "horizon": self.horizon,
            "n_paths": self.paths.shape[0],
            "percentiles": {k: v.tolist() for k, v in self.percentiles.items()},
            "downside_risk": self.downside_risk,
            "scenario_stats": self.scenario_stats,
        }


def simulate_paths(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    dates: pd.Series,
    horizon: str,
    config: Optional[SimulationConfig] = None,
) -> SimulationResult:
    """
    Generate Monte Carlo paths from forecast quantiles.

    Uses a parametric approach: fits a distribution to (P10, P50, P90),
    then samples paths from that distribution.

    Args:
        p10, p50, p90: Forecast quantiles (arrays of same length)
        dates: Corresponding dates
        horizon: Horizon identifier
        config: Simulation configuration

    Returns:
        SimulationResult with paths and risk metrics
    """
    if config is None:
        config = SimulationConfig()

    np.random.seed(config.random_seed)

    n_periods = len(p10)
    logger.info(f"\n{'='*60}")
    logger.info(f"Monte Carlo Simulation - {horizon}")
    logger.info(f"  Periods: {n_periods}")
    logger.info(f"  Paths: {config.n_paths}")
    logger.info(f"{'='*60}")

    # Estimate std from quantiles (assuming normal distribution)
    # P10 ≈ μ - 1.28σ, P90 ≈ μ + 1.28σ
    # So: σ ≈ (P90 - P10) / 2.56
    estimated_std = (p90 - p10) / 2.56

    # Generate paths
    # Each path: sample from N(p50, estimated_std) at each time step
    paths = np.zeros((config.n_paths, n_periods))

    for i in range(config.n_paths):
        # Draw from normal distribution centered at p50 with estimated std
        noise = np.random.normal(loc=0, scale=estimated_std, size=n_periods)
        paths[i, :] = p50 + noise

    # Calculate percentiles across paths
    percentile_levels = [5, 10, 25, 50, 75, 90, 95]
    percentiles = {}
    for p in percentile_levels:
        percentiles[p] = np.percentile(paths, p, axis=0)

    # Calculate downside risk metrics
    final_prices = paths[:, -1]  # Last period prices across all paths

    # Value at Risk (VaR): 5th percentile of final prices
    var_5 = np.percentile(final_prices, 5)

    # Conditional VaR (CVaR): mean of worst 5% outcomes
    worst_5_pct = np.sort(final_prices)[: int(config.n_paths * 0.05)]
    cvar_5 = np.mean(worst_5_pct)

    # Max drawdown across paths
    # For each path, compute max drawdown from initial price
    initial_price = p50[0]
    drawdowns = (paths - initial_price) / initial_price
    max_drawdown = float(np.min(drawdowns))

    downside_risk = {
        "var_5": float(var_5),
        "cvar_5": float(cvar_5),
        "max_drawdown": max_drawdown,
        "downside_probability": float(np.mean(final_prices < initial_price)),
    }

    # Scenario statistics
    scenario_stats = {
        "initial_price": float(initial_price),
        "final_p50": float(p50[-1]),
        "final_mean": float(np.mean(final_prices)),
        "final_std": float(np.std(final_prices)),
        "paths_above_initial": int(np.sum(final_prices > initial_price)),
        "paths_below_initial": int(np.sum(final_prices < initial_price)),
    }

    logger.info("\n📊 Risk Metrics:")
    logger.info(f"  VaR (5%): {var_5:.2f}")
    logger.info(f"  CVaR (5%): {cvar_5:.2f}")
    logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"  Downside Prob: {downside_risk['downside_probability']:.2%}")

    result = SimulationResult(
        horizon=horizon,
        paths=paths,
        percentiles=percentiles,
        downside_risk=downside_risk,
        scenario_stats=scenario_stats,
    )

    logger.info("✅ Monte Carlo simulation complete\n")

    return result


def combine_ensemble_with_mc(
    ensemble_forecasts: Dict[str, EnsembleForecast],
    config: Optional[SimulationConfig] = None,
) -> Dict[str, SimulationResult]:
    """
    Run Monte Carlo on QRA ensemble output for multiple horizons.

    Args:
        ensemble_forecasts: Dict mapping horizon -> EnsembleForecast from QRA
        config: Simulation configuration

    Returns:
        Dict mapping horizon -> SimulationResult
    """
    if config is None:
        config = SimulationConfig()

    results = {}

    for horizon, forecast in ensemble_forecasts.items():
        logger.info(f"\n--- Simulating: {horizon} ---")

        result = simulate_paths(
            p10=forecast.p10,
            p50=forecast.p50,
            p90=forecast.p90,
            dates=forecast.dates,
            horizon=horizon,
            config=config,
        )
        results[horizon] = result

    return results
