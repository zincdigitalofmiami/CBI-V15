"""
Risk and probabilistic metrics for model evaluation.

These utilities are shared across training scripts. They provide:
- Sharpe / Sortino ratios for strategy-style returns.
- Pinball (quantile) loss for probabilistic forecasts.
- Simple helper to compute empirical quantiles.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _to_array(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr[~np.isnan(arr)]


def sharpe_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio.

    Args:
        returns: Sequence of periodic strategy returns (e.g., horizon returns).
        risk_free_rate: Periodic risk‑free rate expressed in same units as returns.
        periods_per_year: Trading periods per year for annualization.
    """
    r = _to_array(returns)
    if r.size == 0:
        return float("nan")

    excess = r - risk_free_rate
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0:
        return float("nan")
    return (mu / sigma) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sortino ratio (downside volatility only).
    """
    r = _to_array(returns)
    if r.size == 0:
        return float("nan")

    excess = r - risk_free_rate
    downside = excess[excess < 0.0]
    if downside.size == 0:
        return float("inf")

    mu = excess.mean()
    sigma_down = downside.std(ddof=1)
    if sigma_down == 0:
        return float("nan")
    return (mu / sigma_down) * np.sqrt(periods_per_year)


def pinball_loss(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    quantile: float,
) -> float:
    """
    Pinball (quantile) loss for a single quantile.

    Args:
        y_true: True values.
        y_pred: Predicted quantile values.
        quantile: Quantile level between 0 and 1 (e.g., 0.5 for median).
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")

    y_t = np.asarray(list(y_true), dtype=float)
    y_p = np.asarray(list(y_pred), dtype=float)
    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    diff = y_t - y_p
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.nanmean(loss))


def empirical_quantiles(
    values: Iterable[float],
    quantiles: Iterable[float],
) -> Tuple[float, ...]:
    """
    Compute empirical quantiles for a 1‑D sample.

    Returns:
        Tuple of quantile values in the same order as `quantiles`.
    """
    v = _to_array(values)
    if v.size == 0:
        return tuple(float("nan") for _ in quantiles)
    q_arr = np.asarray(list(quantiles), dtype=float)
    return tuple(float(x) for x in np.quantile(v, q_arr))


