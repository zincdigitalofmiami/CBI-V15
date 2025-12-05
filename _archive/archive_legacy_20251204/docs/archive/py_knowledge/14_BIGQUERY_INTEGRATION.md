---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**  
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.  
All data must come from authenticated APIs, official sources, or validated historical records.
---

# BigQuery Integration – CBI‑V15 (Storage, Features, Explainability)

**Scope (V15)**: BigQuery is a **storage + ETL + explainability** engine.  
All **training and inference** happens on the **Mac M4**, then results (forecasts, SHAP, Monte‑Carlo summaries, quantiles) are written back to BigQuery for dashboards and audits.

This document replaces legacy V14/BQML/Vertex patterns. It reflects the current V15 design described in:

- `docs/architecture/MASTER_PLAN.md`
- `docs/architecture/TRAINING_PLAN.md`
- `docs/architecture/FINAL_COMPLETE_BQ_SCHEMA.sql`
- `docs/reference/COMPLETE_SYSTEM_FLOW.md`
- `docs/reference/COMPLETE_FEATURE_LIST_290.md`

---

## 1. Datasets & Roles (V15)

From `FINAL_COMPLETE_BQ_SCHEMA.sql` and MASTER_PLAN:

- `market_data` – Venue‑pure Databento futures OHLCV (1m, 1d) and any historical bridge.  
- `raw_intelligence` – External “slow” data (FRED, USDA, EIA, CFTC, NOAA, classified news, sentiment).  
- `signals` – Derived daily signals (crush, spreads, Big‑8, hidden relationships).  
- `features` – Canonical wide feature tables (e.g. `features.master_features`, `features.daily_ml_matrix`).  
- `training` – Frozen training tables per horizon/asset (`zl_training_prod_allhistory_*`, MES tables if used).  
- `regimes` / `training.regime_*` – Regime calendar + weights.  
- `neural` / `drivers` – Optional neural feature and driver tables (if/when wired).  
- `predictions` – Forecast tables and quantile/Monte‑Carlo summaries.  
- `monitoring` – Model/OOS metrics, drift, MAPE/Sharpe, alert substrates.  
- `dim` / `ops` – Reference metadata and operational audit tables (e.g. ingestion completion).

**Key rule**: BigQuery owns **data contracts and history**, not training logic.

---

## 2. End‑to‑End Flow (High Level)

1. **Ingest → BigQuery**
   - Python scripts in `src/ingestion/` pull from Databento, FRED, USDA, EIA, CFTC, NOAA, ScrapeCreators, etc.
   - They write **directly** into `market_data.*` and `raw_intelligence.*` (and optionally Parquet mirrors on the external drive).

2. **ETL & Features → BigQuery (Dataform/SQL)**
   - Dataform/SQL transforms RAW into staging‑equivalent and wide feature tables:
     - Rolling correlations, spreads, regimes, sentiment aggregates, weather buckets, CFTC positioning.
     - Pivot pattern only (no train‑time joins): one row per date (and symbol where applicable).
   - Output: `features.master_features` / `features.daily_ml_matrix` and Big‑8/B8+ driver tables in `signals.*`.

3. **Training Freeze → BigQuery → Mac**
   - Freeze training tables in `training.zl_training_prod_allhistory_{horizon}` (and optional `training.zl_training_full_allhistory_*` for research).
   - Export to Parquet (via `scripts/export_training_data.py`) into `TrainingData/exports/`.
   - Mac M4 training code reads these Parquet files; **no BQML, no cloud training**.

4. **Forecasts + SHAP + Monte‑Carlo → BigQuery**
   - Mac generates:
     - Point forecasts and **quantile bands** per horizon (e.g. p10/p50/p90 or 90% intervals).
     - SHAP per feature per date/horizon (TreeSHAP on ensembles).
     - Monte‑Carlo summaries (cone quantiles, scenario surfaces) from local simulations.
   - These are written back into `predictions.*` and `models_v4.shap_daily` (or V15‑equivalent tables).

5. **Dashboards → BigQuery Views → Vercel**
   - Thin views in `predictions`/`api` join predictions, quantiles, SHAP, regimes and Big‑8 signals into small JSON‑friendly shapes.
   - Next.js/Vercel reads from those views only; no direct table writes, no model logic in the dashboard.

---

## 3. Feature & Training Contracts

### 3.1 Wide Feature Tables

**Goal**: Never join at train time.

- `features.master_features` / `features.daily_ml_matrix`:
  - One row per `(date, symbol)` (ZL primary).  
  - Columns grouped roughly as in `COMPLETE_FEATURE_LIST_290.md`:
    - Price/volume and lags.  
    - Big‑8 drivers and extensions (crush, China, dollar, Fed, tariffs, biofuels, crude, VIX, positioning).  
    - Cross‑asset correlations and betas (palm, crude, FX, grains).  
    - Weather and ag indices (Brazil/Argentina/US).  
    - CFTC positioning.  
    - Technical indicators (trend, vol, structure).  
    - Sentiment/news buckets + 9‑layer sentiment architecture roll‑ups.  
  - Partitioned by `date`, clustered by `symbol` (and where needed, `bucket`/`regime`).

### 3.2 Training Tables

From `FINAL_COMPLETE_BQ_SCHEMA.sql`:

- `training.zl_training_prod_allhistory_1w` (canonical schema; 1w horizon).  
- Copy‑schema tables for `1m/3m/6m/12m`: `training.zl_training_prod_allhistory_{h}`.  
- Fields (conceptually):
  - `date`, `regime`, `training_weight`, horizon‑specific `target_{h}`, plus all selected features.  
  - `as_of` timestamp for freeze auditability.
- Optional “full” research tables (`training.zl_training_full_allhistory_*`) with 1,900+ columns.

**Freeze discipline**:
- Training freezes are **time‑cutoff snapshots** (no future info), created by SQL/Dataform with explicit `as_of` and embargo rules.
- Mac training scripts must treat these tables as read‑only.

---

## 4. Forecasts, Quantiles & Monte‑Carlo in BigQuery

### 4.1 Forecast Tables

**Schema pattern** (per horizon, e.g. `predictions.zl_predictions_1m`):

- Keys: `date`, `horizon`, `symbol` (ZL)  
- Core:
  - `prediction` (point forecast).  
  - `p10`, `p50`, `p90` (or `prediction_lower_90`, `prediction_median`, `prediction_upper_90`).  
  - `confidence_score` or calibrated coverage metric.  
  - `model_type`, `model_version`, `as_of`.  
- Optional:
  - Regime tags, shock flags, feature counts, training window metadata.

**Usage**:
- Cone charts, confidence bands, probability‑of‑breach calculations in the dashboard.

### 4.2 Monte‑Carlo Surfaces

Monte‑Carlo runs live on the Mac (block‑bootstrap / regime‑aware simulations) and pushes **summaries** to BigQuery:

- Either:
  - Extra columns in predictions tables (`mc_p10`, `mc_p50`, `mc_p90`), or  
  - Separate tables (`predictions.zl_mc_cones_{h}`) with:
    - `date`, `horizon`, `days_ahead`, `p10`, `p25`, `p50`, `p75`, `p90`, `scenario_id?`.

BigQuery does **not** run Monte‑Carlo; it stores the **aggregated path statistics** used for:

- Monte‑Carlo pinball charts (cone width, quantile band animation).  
- Scenario overlays (“policy shock”, “biofuel shock”, “La Niña severity”) via pre‑computed scenario tables.

---

## 5. SHAP & Explainability in BigQuery

From MASTER_PLAN’s SHAP section:

- SHAP is computed **nightly on the Mac** using TreeSHAP on the ensemble models.  
- Raw SHAP output is converted to **¢/lb contribution** per feature, per date, per horizon.  
- Stored in a long, additive table (e.g. `models_v4.shap_daily`):
  - `date`, `horizon`, `symbol`, `feature_name`, `shap_value_cents`, `as_of`.

BigQuery’s role:

- Join SHAP with predictions and regimes to back each forecast with a full factor breakdown.  
- Power SHAP overlays and waterfalls in the dashboard via views:
  - Top N drivers per date/horizon.  
  - Time‑series of SHAP contributions for key features (RINs, drought, tariffs, FX, etc.).  
- Provide a **100% additive audit trail** of every forecasted penny.

---

## 6. Plugin Libraries & Enhancement Pattern

V15 allows “plugin” quant libraries **off‑BigQuery**, then feeds their outputs back as features or diagnostic tables:

- **QuantLib / risk libs** – derivatives pricing, term structures, Monte‑Carlo payoffs.  
  - Input: historical prices/curves exported from BigQuery.  
  - Output: risk metrics, convexity, optional “risk premium” features stored under `features.*` or `signals.*`.

- **Nixtla (statsforecast/neuralforecast)** – advanced TS models used as baselines or specialist learners.  
  - Input: training tables exported from BigQuery.  
  - Output: forecasts/quantiles injected into `predictions.*` or used to ensemble with core models.

- **arch / volatility tools** – GARCH/ARCH volatility estimates.  
  - Output: realized/forecast vol features added to `features.*` and used in Monte‑Carlo / cone scaling.

- **vectorbt / backtesting** – strategy backtests on exported data.  
  - Output: evaluation metrics under `monitoring.*` or external experiment logs.

**Contract**:

- All plugins read from **exported BigQuery data** (Parquet or direct query).  
- Any useful output must be written back into clearly named tables/columns:
  - `features.plugin_*`, `signals.plugin_*`, `predictions.plugin_*`, `monitoring.plugin_metrics_*`.  
- Plugins never write arbitrary tables or mutate RAW; they’re **append‑only enhancements**.

---

## 7. Practical Integration Checklist (V15)

When wiring new components, follow this checklist:

1. **Ingest**  
   - [ ] New source lands in `market_data` or `raw_intelligence` with clear prefixes, partitioned by date.
   - [ ] Ingestion status recorded in `ops.ingestion_completion`.

2. **Features**  
   - [ ] Features derived into `features.*` or `signals.*` via Dataform/SQL.  
   - [ ] Wide tables only; train‑time requires no joins.

3. **Training**  
   - [ ] Training freezes written to `training.zl_training_prod_allhistory_*` with `regime`, `training_weight`, `as_of`.  
   - [ ] Exports to `TrainingData/exports/` for Mac training.

4. **Forecasts & Quantiles**  
   - [ ] Mac writes predictions + quantile bands into `predictions.zl_predictions_*`.  

5. **SHAP & Monte‑Carlo**  
   - [ ] SHAP daily table updated (`models_v4.shap_daily` or V15 equivalent).  
   - [ ] Monte‑Carlo summaries stored as quantile cones (per date/horizon).

6. **Dashboards**  
   - [ ] API views read only from `predictions`, `features`, `signals`, `regimes`, SHAP tables.  
   - [ ] No model logic in SQL; SQL is projection/aggregation only.

This is the **current** V15 BigQuery integration contract; any legacy V14 BQML/Vertex/forecasting_data_warehouse patterns are deprecated and should not be used as design inputs.  

