---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**  
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.  
All data must come from authenticated APIs, official sources, or validated historical records.
---

**Best Practices**  
For mandatory rules (data quality, us‑central1 only, Mac‑only training, no hard‑coded keys), see:  
`docs/reference/BEST_PRACTICES.md`

# CBI‑V15 Training Plan – Mac M4 Only

**Last Updated**: November 28, 2025  
**Status**: Architecture defined, baselines pending data ingestion  
**Architecture**: Local‑first (Mac M4), no BQML, no Vertex AI  
**Data Plane**: BigQuery for storage, ETL, explainability only  

This document is the **authoritative training spec** for V15. It replaces earlier V14/BQML/Vertex dual‑track descriptions.

MASTER_PLAN.md covers overall architecture. This file goes deep on **training phases**, **model families**, and the **ensemble/prediction pipeline**.

---

## 1. Scope & Goals

**Primary target**: ZL (soybean oil futures), daily bars, horizons:
- 1w, 1m, 3m, 6m, 12m (daily forecast horizons).

**Objectives**:
- Use **25 years of data** (2000–2025) with regime‑aware weighting.  
- Run all training and inference on **Mac M4**.  
- Keep BigQuery as **source‑of‑truth storage**, **feature assembly**, and **explainability sink** (forecasts, SHAP, Monte‑Carlo summaries).

---

## 2. BigQuery Training Contracts

From `FINAL_COMPLETE_BQ_SCHEMA.sql` and MASTER_PLAN:

- **Features**:
  - `features.master_features` / `features.daily_ml_matrix` – wide, pivoted feature tables (no joins at train time).

- **Training tables (frozen)**:
  - `training.zl_training_prod_allhistory_1w`
  - `training.zl_training_prod_allhistory_1m`
  - `training.zl_training_prod_allhistory_3m`
  - `training.zl_training_prod_allhistory_6m`
  - `training.zl_training_prod_allhistory_12m`

  Columns (conceptual):
  - `date`, `symbol` (ZL), `regime`, `training_weight`, `as_of`.  
  - Horizon‑specific target (`target_1w`, `target_1m`, …).  
  - Selected feature set (curated from `features.*`).

- **Regimes & weights**:
  - `training.regime_calendar` – maps dates to regime labels.  
  - `training.regime_weights` – 50–5000 weight scale per regime (recency + relevance).

- **Predictions & explainability**:
  - `predictions.zl_predictions_{h}` – point forecasts + quantiles per horizon.  
  - `models_v4.shap_daily` (or V15 equivalent) – SHAP per feature/date/horizon in cents/lb.  
  - Optional Monte‑Carlo cones (`predictions.zl_mc_cones_{h}`) – quantile paths from simulations.

These tables are the **only contracts** the Mac training pipeline depends on.

---

## 3. Training Phases (End‑to‑End)

We run training as a repeatable, leak‑proof pipeline with four phases:

1. **Phase 0 – Data readiness & freeze**  
2. **Phase 1 – Baseline model family** (slot #1)  
3. **Phase 2 – Three additional model families** (slots #2–#4)  
4. **Phase 3 – Ensemble & prediction pipeline**

Monitoring/re‑training cadence sits on top of these phases.

---

## 4. Phase 0 – Data Readiness & Freeze

**Goal**: Ensure training tables are complete, well-formed, and leak‑free before touching any model.

**Inputs**:
- `features.master_features` / `features.daily_ml_matrix`  
- `training.regime_calendar`, `training.regime_weights`

**Steps**:
- **0.1 Ingestion & feature checks**
  - Verify ingestion completion in `ops.ingestion_completion` for Databento, FRED, USDA, CFTC, EIA, NOAA, news.  
  - Run Dataform/SQL to rebuild features up to `as_of` cutoff date.

- **0.2 Freeze training tables**
  - For each horizon `h` in {1w, 1m, 3m, 6m, 12m}:
    - Create/refresh `training.zl_training_prod_allhistory_{h}` with:
      - All features up to cutoff date (no future info).  
      - Regime label + `training_weight` from regime tables.  
      - Target column `target_{h}` defined as future price level (no returns leakage).  
      - `as_of` timestamp.

- **0.3 Basic validation**
  - No null targets.  
  - No future dates beyond cutoff.  
  - Regime coverage matches calendar.  
  - Reasonable row counts (full 2000–cutoff coverage, per MASTER_PLAN).

Exports to Mac:
- Use `scripts/export_training_data.py` to write Parquet files into `TrainingData/exports/zl_training_{h}.parquet`.

---

## 5. Phase 1 – Baseline Model Family (Slot #1)

**Family #1**: Robust tree‑based baseline for each horizon.

Recommended:
- **Model**: LightGBM (or CatBoost) per horizon, trained on the curated feature subset.  
- **Loss**: MAE / pinball loss (for quantiles) per horizon.  
- **Features**: Core price/volume, Big‑8 drivers, selected cross‑asset, weather, sentiment, CFTC.

Per horizon `h`:
- Load `TrainingData/exports/zl_training_{h}.parquet` on the Mac.  
- Split into train/val/test by date (no shuffling, horizon‑specific splits).  
- Train baseline model with:
  - Regime weights (weight column).  
  - Basic hyperparameter tuning (Optuna/hand‑tuned).  
  - Output:
    - Point forecasts.  
    - Quantile forecasts (e.g. p10/p50/p90).

**Artifacts**:
- Saved model: `Models/local/baseline/zl_{h}_lightgbm.pt` (or equivalent).  
- Metrics: written to a local log and pushed to `monitoring.*` as needed.

This baseline becomes the **reference yardstick** for any additional model family.

---

## 6. Phase 2 – Three Additional Model Families (Slots #2–#4)

We add three more families, each with a clear purpose and regime role.

### 6.1 Family #2 – Statistical / Classical TS

Purpose: Capture long‑memory and level dynamics; provide sanity‑check baselines.

Options:
- ARIMA/ETS/Theta via `statsforecast`.  
- Simple Kalman/BSTS models for trend + seasonal components.

Usage:
- Train per horizon (or per target series) using the same `zl_training_{h}` data, possibly only on a subset of features (price, curves, vol).

### 6.2 Family #3 – Deep Time‑Series Models

Purpose: Capture non‑linear temporal structure across multiple features and regimes.

Options:
- TFT / N‑HiTS / NHITSx / TemporalConvNet via Nixtla `neuralforecast` or custom PyTorch.  
- Input: long panels built from `zl_training_{h}` (reshaped for sequence models).

Usage:
- Train per horizon with strict time‑based validation; produce point and quantile forecasts.  
- Use M4’s GPU (MPS) with mixed precision for efficiency.

### 6.3 Family #4 – Regime‑Specialist Models

Purpose: Exploit regime heterogeneity (trend vs mean‑revert vs shock).

Patterns:
- Separate models per regime:  
  - **Trend**: slower features, momentum heavy.  
  - **Mean‑revert**: z‑scores, spread deviation, half‑life features.  
  - **Shock**: high‑vol and event features, jump indicators.
- Or shared model with regime inputs and **regime‑conditioned loss/weights**.

Usage:
- Train only on rows belonging to a given regime (or with high regime weights).  
- Outputs are blended later by the ensemble/meta‑switch.

---

## 7. Phase 3 – Ensemble & Prediction Pipeline

Once all four slots (baseline + 3 families) are trained, we build the final prediction and write it back to BigQuery.

### 7.1 Meta‑Model / Ensemble

Inputs per horizon `h`:
- Baseline forecasts + quantiles.  
- Statistical model forecasts.  
- Deep model forecasts.  
- Regime‑specialist forecasts.  
- Optional: recent OOS errors per model, regime probabilities.

Strategies:
- Simple weighted average with weights based on recent OOS performance & regime fit.  
- Or a small meta‑learner (e.g. linear/elastic net) trained only on OOS folds.

Outputs:
- Final point forecast and quantile bands per horizon/date.  
- Model contribution weights (for monitoring).

### 7.2 Writing Predictions to BigQuery

Per horizon `h`:
- Construct a DataFrame with:
  - `date`, `symbol` (ZL), `horizon`.  
  - `prediction`, `p10`, `p50`, `p90` (or equivalent).  
  - `confidence_score`, `model_type`, `model_version`, `as_of`.  
- Use `scripts/upload_predictions.py` to write into `predictions.zl_predictions_{h}`.

### 7.3 SHAP & Monte‑Carlo

On the Mac:
- Run TreeSHAP on the ensemble (or primary tree‑based family) to compute per‑feature SHAP values in cents/lb.  
- Aggregate Monte‑Carlo simulations into cone quantiles per horizon/date (block‑bootstrap or regime‑aware).

Write back to BigQuery:
- SHAP: long table (`models_v4.shap_daily` or V15 equivalent).  
- Monte‑Carlo: quantile cones (`predictions.zl_mc_cones_{h}` or extra columns on predictions tables).

These tables power SHAP waterfalls, factor breakdowns, and cone charts in the dashboard.

---

## 8. Monitoring & Retraining

Monitoring is not optional; every promotion cycle must be justified.

**Metrics**:
- Horizon‑specific MAPE/RMSE, bias, hit rates, regime‑conditional performance.  
- Coverage of quantile bands (e.g. p10–p90 coverage near 80–90%).  
- Stability of SHAP attribution over time.

**Cadence**:
- Daily/weekly evaluation jobs on Mac summarizing performance and pushing metrics into `monitoring.*`.  
- Periodic retraining (e.g. monthly or on drift detection).

**Promotion rules**:
- Only promote a new ensemble if:
  - OOS performance is **not worse** than current by agreed thresholds.  
  - No leak is detected (respect embargo, no future info).  
  - Training tables and regimes pass basic validation.

---

## 9. How This Doc Connects to MASTER_PLAN

- MASTER_PLAN.md defines the **system architecture** and the role of BigQuery/Dataform/Mac/Vercel.  
- This TRAINING_PLAN.md defines the **training side**:
  - BigQuery training contracts.  
  - Phases from freeze to ensemble.  
  - Model families (baseline + three others).  
  - How predictions, SHAP, and Monte‑Carlo results are written back for the dashboard.

Any future changes to the training stack should update this file and keep it in sync with MASTER_PLAN and `FINAL_COMPLETE_BQ_SCHEMA.sql`.
