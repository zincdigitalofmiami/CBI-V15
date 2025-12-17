# Ensemble Module (L3)

## Purpose

Regime-weighted Quantile Regression Averaging (QRA) for combining model outputs.

This is **Level 3** in the legacy modeling stack:

- L1: Base models generate quantile forecasts
- L2: Meta-learner selects candidates
- **L3: QRA combines forecasts** ← YOU ARE HERE (experimental/legacy)
- L4: Monte Carlo simulates risk

## What Belongs Here

- `qra_ensemble.py` - Quantile Regression Averaging implementation
- Future ensemble methods:
  - `bayesian_model_averaging.py`
  - `stacking_ensemble.py`
  - `dynamic_model_averaging.py`

## Current Implementation

### QuantileForecast (Dataclass)

Represents a single model's quantile forecast:

```python
@dataclass
class QuantileForecast:
    model_name: str
    horizon: str
    dates: pd.Series
    p10: np.ndarray  # 10th percentile
    p50: np.ndarray  # 50th percentile (median)
    p90: np.ndarray  # 90th percentile
```

### EnsembleForecast (Dataclass)

QRA ensemble result:

```python
@dataclass
class EnsembleForecast:
    horizon: str
    dates: pd.Series
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    weights: Dict[str, float]  # Model weights used
    regime: str
```

### run_qra() Function

Main QRA implementation:

```python
def run_qra(
    forecasts: List[QuantileForecast],
    weights: Dict[str, float],
    regime: str = "adaptive",
) -> EnsembleForecast
```

**Features:**

- Weighted average of quantiles (NOT simple point forecast averaging)
- Preserves full uncertainty structure
- Regime-aware weights (suggested by AutoGluon, executed here)

## Usage

```python
from src.ensemble import run_qra, QuantileForecast

# Create quantile forecasts from models
forecasts = [
    QuantileForecast(
        model_name="catboost",
        horizon="1w",
        dates=dates,
        p10=catboost_p10,
        p50=catboost_p50,
        p90=catboost_p90,
    ),
    QuantileForecast(
        model_name="lightgbm",
        horizon="1w",
        dates=dates,
        p10=lgbm_p10,
        p50=lgbm_p50,
        p90=lgbm_p90,
    ),
]

# Define weights (from AutoGluon or validation performance)
weights = {
    "catboost": 0.6,
    "lightgbm": 0.4,
}

# Run QRA
ensemble = run_qra(
    forecasts=forecasts,
    weights=weights,
    regime="high_volatility",
)

# Access results
ensemble_df = ensemble.to_dataframe()
# Columns: date, forecast_p10, forecast_p50, forecast_p90, horizon, regime
```

## Integration Notes

- Historically, AutoGluon's **ForecasterAgent** called QRA:
  1. Gathered quantile forecasts from L1/L2 models
  2. Asked OpenAI for weight suggestions (based on regime)
  3. Executed QRA numerically (this module)
  4. Passed results to L4 (Monte Carlo)

- In V15.1, the **canonical ensemble** is AutoGluon’s `WeightedEnsemble_L2`.  
  This QRA module is retained as an optional/legacy ensemble experiment, not the primary production path.

## Metrics

**Interval Score:** Evaluates probabilistic forecast quality

```python
from src.ensemble import calculate_interval_score

score = calculate_interval_score(
    actual=actual_prices,
    p10=ensemble_p10,
    p90=ensemble_p90,
    alpha=0.1,
)
# Lower is better; penalizes width and coverage violations
```

## What Does NOT Belong Here

- Base model training (→ `src/training/` / AutoGluon wrappers, or legacy baselines)
- Model selection / orchestration logic (→ AutoGluon, legacy AutoGluons, or scripts)
- Risk simulation (→ `src/simulators/`)

## Philosophy

QRA should be:

1. Purely numeric (no LLM logic here)
2. Preserve quantile structure
3. Regime-aware (via weights)
4. Fast (< 1 second for typical ensemble)

---

**For L4 (Monte Carlo), see:** `src/simulators/README.md`
