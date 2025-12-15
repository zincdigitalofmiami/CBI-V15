# Baselines

## Purpose
Baseline model implementations - simple, interpretable models that set performance benchmarks **outside** the canonical V15.1 AutoGluon stack.

These scripts are **legacy / experimental baselines**. The production modeling stack for V15.1 is:
- Big 8 bucket specialists → AutoGluon `TabularPredictor`
- Core ZL forecaster → AutoGluon `TimeSeriesPredictor`
- Meta + ensemble → AutoGluon stacking + `WeightedEnsemble_L2`
- Monte Carlo → consumes final forecasts only

Use these baselines only for offline comparison and debugging, not as the primary training pipeline.

## What Belongs Here
- `lightgbm_zl.py` - LightGBM baseline (point forecasts, legacy)
- `catboost_zl.py` - CatBoost quantile regression (P10/P50/P90, legacy)
- `xgboost_zl.py` - XGBoost quantile regression (P10/P50/P90, legacy)
- Future (if ever added): `tft_zl.py`, `prophet_zl.py`, `garch_zl.py` as **non-canonical experiments only**

## Current Models

### 1. LightGBM (`lightgbm_zl.py`) – legacy baseline
- **Type:** Gradient Boosting (point forecast)
- **Horizons:** 1w, 1m, 3m, 6m, 12m
- **Output:** P50 (median) predictions
- **Use:** Fast baseline, good for comparison

### 2. CatBoost (`catboost_zl.py`) – legacy baseline
- **Type:** Gradient Boosting (quantile regression)
- **Horizons:** 1w, 1m, 3m, 6m, 12m
- **Output:** P10/P50/P90 (probabilistic)
- **Metrics:** Pinball loss, coverage, MAE
- **Use:** Primary quantile forecaster

### 3. XGBoost (`xgboost_zl.py`) – legacy baseline
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
3. Generate probabilistic forecasts (quantiles) where applicable
4. Serve strictly as benchmarks for the **AutoGluon-based** Big 8 + ZL pipeline, not replace it

## Naming Convention
`{model_family}_{asset}.py`

Example: `catboost_zl.py`

## Usage

```bash
# Train individual model
python src/training/baselines/catboost_zl.py
```

## Integration Notes

- These baselines can be invoked directly from CLI or orchestration scripts.
- AutoGluon-based orchestration has been removed; use the AutoGluon-centric pipeline as primary, and treat these as optional benchmarks only.
