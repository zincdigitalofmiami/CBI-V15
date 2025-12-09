# Baselines

## Purpose
Baseline model implementations - simple, interpretable models that set performance benchmarks.

Baselines are L1 in the modeling stack: they generate quantile forecasts (P10/P50/P90)
which are then combined by L3 (QRA ensemble) and risk-tested by L4 (Monte Carlo).

## What Belongs Here
- `lightgbm_zl.py` - LightGBM baseline (point forecasts)
- `catboost_zl.py` - CatBoost quantile regression (P10/P50/P90)
- `xgboost_zl.py` - XGBoost quantile regression (P10/P50/P90)
- Future: `tft_zl.py`, `prophet_zl.py`, `garch_zl.py`

## Current Models

### 1. LightGBM (`lightgbm_zl.py`)
- **Type:** Gradient Boosting (point forecast)
- **Horizons:** 1w, 1m, 3m, 6m, 12m
- **Output:** P50 (median) predictions
- **Use:** Fast baseline, good for comparison

### 2. CatBoost (`catboost_zl.py`)
- **Type:** Gradient Boosting (quantile regression)
- **Horizons:** 1w, 1m, 3m, 6m, 12m
- **Output:** P10/P50/P90 (probabilistic)
- **Metrics:** Pinball loss, coverage, MAE
- **Use:** Primary quantile forecaster

### 3. XGBoost (`xgboost_zl.py`)
- **Type:** Gradient Boosting (quantile regression)
- **Horizons:** 1w, 1m, 3m, 6m, 12m
- **Output:** P10/P50/P90 (probabilistic)
- **Metrics:** Pinball loss, coverage, MAE
- **Use:** Alternative to CatBoost, often faster

## Training Data

All models expect training data from:
```
training.daily_ml_matrix_zl
```

With columns (subset):
- `as_of_date`, `symbol` (keys)
- 300+ features (from Anofox)
- `target_ret_1w`, `target_ret_1m`, `target_ret_3m`, `target_ret_6m`, `target_ret_12m` (targets)
- `train_val_test_split` (train/val/test)

## Model Outputs

Models save artifacts to:
```
models/baselines/{family}/
  zl_1w_p10.{ext}
  zl_1w_p50.{ext}
  zl_1w_p90.{ext}
  ...
```

## Philosophy
Baselines should be:
1. Simple and interpretable
2. Fast to train (< 30 min on Mac M4)
3. Generate probabilistic forecasts (quantiles)
4. Serve as benchmarks for complex models

## Naming Convention
`{model_family}_{asset}.py`

Example: `catboost_zl.py`

## Usage

```bash
# Train individual model
python src/training/baselines/catboost_zl.py

# Or let TSci orchestrate
python src/models/tsci/planner.py
```

## Integration with TSci

TSci agents orchestrate these baselines via:
- `src/models/tsci/model_sweep.py` - Runs multi-model sweeps
- `src/engines/engine_registry.py` - Model family registry
- `src/models/tsci/forecaster.py` - Ensemble + risk pipeline

