# MotherDuck Migration & 45-Model Ensemble Plan

**Status:** LOCKED  
**Date:** December 3, 2025  
**Scope:** 100% BigQuery replacement with MotherDuck + AnoFox + Neural + Meta-Ensemble

---

## Executive Summary

**Goal:** Replace BigQuery entirely with MotherDuck + 45-model forecasting stack for ZL soybean oil procurement.

**Model Universe:**
- **Tier 1:** 31 AnoFox statistical models (SQL-native baseline)
- **Tier 2:** 7 neural models (PyTorch)
- **Tier 3:** 7 meta-ensemble models (hybrid + stacking)

**Production Ensemble:** 5-7 active models per horizon, regime-aware weights

**Timeline:** 25 days (5 phases)

---

## Schema Contract (8 Datasets - LOCKED)

All work confined to these datasets in MotherDuck:

| Dataset | Purpose | AnoFox Usage | Neural Usage |
|---------|---------|--------------|--------------|
| `raw` | Vendor data as-delivered | Source data | Read-only |
| `raw_staging` | Per-run temp (`<source>_<bucket>_<run_id>`) | Write first, then MERGE | N/A |
| `staging` | Normalized daily panels | Model inputs | Training data source |
| `features` | Engineered features (276 → 80-120 pruned) | Feature importance | Feature selection |
| `training` | `daily_ml_matrix` canonical | Primary input | Primary input |
| `reference` | Driver groups, model registry, regime calendar | Read regimes | Read regimes |
| `signals` | Live scores, ensemble output | Write forecasts | Write neural forecasts |
| `ops` | QC, tests, logs, pipeline metrics | Write QC results | Write training logs |

**NO NEW SCHEMAS. NO EXCEPTIONS.**

---

## Column Prefix Contract (RUTHLESS)

**Unprefixed:** `date`, `symbol` ONLY

**All others prefixed:**
- FRED: `fred_*`
- Databento: `databento_*`
- Weather: `weather_{country}_{region}_{variable}`
- EIA: `eia_*`
- USDA: `usda_*`
- CFTC: `cftc_*`
- ScrapeCreators: `scrc_*`
- Policy: `policy_trump_*`
- FX: `fx_*`
- Technical: `tech_*` (only if already exists in 276)

---

## Phase 1: MotherDuck Setup (Days 1-2)

### 1.1 Account & Database
- Create MotherDuck account
- Create database: `usoil_intelligence`
- Configure Vercel env: `MOTHERDUCK_TOKEN`

### 1.2 Schema Creation
Execute DDL to create 8 schemas with tables:

**raw schema:**
- `raw.databento_futures_ohlcv_1d` (PRIMARY KEY: symbol, date)
- `raw.fred_economic` (PRIMARY KEY: series_id, date)
- `raw.scrapecreators_news_buckets`
- `raw.weather_noaa`, `raw.cftc_cot`, `raw.usda_wasde`, `raw.eia_biofuels`

**staging schema:**
- `staging.market_daily` (joined daily panel)
- `staging.zl_prices_clean` (AnoFox-cleaned)
- `staging.fred_macro_panel`
- `staging.weather_granular_daily`

**features schema:**
- `features.zl_features` (276 engineered features)
- `features.feature_importance_log` (AnoFox analysis results)

**training schema:**
- `training.daily_ml_matrix` (canonical 276-feature matrix)
- `training.model_backtest_zl` (backtest results)
- `training.zl_model_runs` (all 45 model training logs)

**forecast schema:**
- `forecast.zl_anofox_panel` (31 statistical baseline forecasts)
- `forecast.zl_neural_panel` (7 neural forecasts)
- `forecast.zl_ensemble_output` (final production forecast)

**reference schema:**
- `reference.driver_group` (8 groups: BIOFUEL_ENERGY, TRADE_TARIFF, etc.)
- `reference.feature_to_driver_group_map` (276 features → driver groups)
- `reference.model_registry` (active model metadata)
- `reference.regime_calendar` (historical regime assignments)

**signals schema:**
- `signals.big_eight_live` (15-minute driver scores)
- `signals.driver_group_score_daily` (daily driver contributions)
- `signals.driver_group_contribution` (SHAP aggregations)

**ops schema:**
- `ops.data_quality_zl_daily` (QC results)
- `ops.pipeline_test_runs` (test logs)
- `ops.ingestion_completion` (data load logs)
- `ops.pipeline_metrics_daily` (composite metrics)

### 1.3 AnoFox Extension Installation
```sql
INSTALL anofox_forecast FROM community;
INSTALL anofox_tabular FROM community;
INSTALL anofox_statistics FROM community;
LOAD anofox_forecast;
LOAD anofox_tabular;
LOAD anofox_statistics;
```

Verify installation:
```sql
SELECT * FROM anofox_forecast_version();
```

---

## Phase 2: Data Migration (Days 3-5)

### 2.1 BigQuery Export (Use Existing Script)
- Script: `scripts/migration/export_bq_to_parquet_v2.py`
- Target: 46 tables, ~3.7M rows
- Storage: `/Volumes/Satechi Hub/CBI-V15/Data/parquet/`
- Preserve: All partitioning metadata

### 2.2 Feature Migration
- Export `features.master_features` (276 features)
- Verify column prefixes intact
- Load to MotherDuck `features.zl_features`

### 2.3 Historical Data Load
- ZL prices: 2000-present (25 years)
- FRED: 55+ series
- Weather: US Midwest, Brazil, Argentina
- CFTC: COT positioning
- USDA: WASDE, exports
- EIA: Biofuels, RINs

### 2.4 Idempotent Load Pattern
```sql
-- 1. Load to staging
CREATE TEMP TABLE raw_staging.zl_prices_20251203 AS 
SELECT * FROM read_parquet('/path/to/zl_prices.parquet');

-- 2. MERGE to canonical
MERGE INTO raw.databento_futures_ohlcv_1d AS T
USING raw_staging.zl_prices_20251203 AS S
ON T.symbol = S.symbol AND T.date = S.date
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...;

-- 3. Log completion
INSERT INTO ops.ingestion_completion (source, start_date, end_date, row_count, status)
VALUES ('databento', '2000-01-01', '2025-12-03', 6543, 'success');
```

---

## Phase 3: AnoFox "Big Run" - The Baseline (Days 6-9)

### 3.1 Data Quality (AnoFox Tabular)

**Gap Detection:**
```sql
CREATE TABLE ops.data_quality_zl_daily AS
SELECT 
    'zl_prices' AS table_name,
    CURRENT_DATE AS report_date,
    *
FROM TS_DETECT_GAPS('staging.market_daily', 'date', 'DAY')
WHERE symbol = 'ZL';
```

**Outlier Detection:**
```sql
INSERT INTO ops.data_quality_zl_daily
SELECT 
    'zl_prices' AS table_name,
    CURRENT_DATE,
    *
FROM TS_OUTLIER_DETECT('staging.market_daily', 'close', method := 'zscore', threshold := 3.0)
WHERE symbol = 'ZL';
```

**Data Prep:**
```sql
CREATE TABLE staging.zl_prices_clean AS
SELECT * FROM TS_FILL_GAPS('staging.market_daily', 'date', '1 DAY')
WHERE symbol = 'ZL' 
  AND anofox_outlier_score(close) < 3.0;
```

### 3.2 The Big Run: 31 Models × 3 Engines × 5 Horizons

**ZL Price Engine (31 models):**
```sql
CREATE TABLE forecast.zl_anofox_panel AS
SELECT 
    'price' AS engine_type,
    model_name,
    forecast_step,
    date_col,
    point_forecast,
    lower_95,
    upper_95
FROM TS_FORECAST_BY(
    'staging.zl_prices_clean',
    group_cols := NULL,
    date_col := 'date',
    value_col := 'close',
    models := ['Naive', 'SeasonalNaive', 'SeasonalWindowAverage', 'RandomWalkDrift',
               'ETS', 'AutoETS', 'Holt', 'HoltWinters', 
               'ARIMA', 'AutoARIMA', 'Theta',
               'TBATS', 'MSTL', 'MFLES', 'STL',
               'Croston', 'CrostonOptimized', 'CrostonSBA', 'ADIDA', 'IMAPA', 'TSB',
               -- All 31 models listed explicitly
              ],
    horizons := [5, 21, 63, 126, 252],  -- 1W, 1M, 3M, 6M, 12M
    confidence_level := 0.90
);
```

**ZL Volatility Engine (GARCH + variants):**
```sql
INSERT INTO forecast.zl_anofox_panel
SELECT 
    'volatility' AS engine_type,
    model_name,
    *
FROM TS_FORECAST_BY(
    'staging.zl_realized_vol',
    group_cols := NULL,
    date_col := 'date',
    value_col := 'realized_vol_21d',
    models := ['GARCH', 'AutoETS', 'ARIMA'],  -- Vol-specific subset
    horizons := [5, 21, 63, 126, 252]
);
```

**Macro Baseline Engine (Slow-moving indicators):**
```sql
INSERT INTO forecast.zl_anofox_panel
SELECT 
    'macro' AS engine_type,
    model_name,
    *
FROM TS_FORECAST_BY(
    'staging.fred_macro_panel',
    group_cols := 'series_id',
    date_col := 'date',
    value_col := 'value',
    models := ['ETS', 'AutoETS', 'Holt', 'TBATS', 'MSTL'],  -- Macro-specific subset
    horizons := [21, 63, 126, 252]  -- Skip 1W for macro
);
```

### 3.3 Baseline Evaluation
```sql
CREATE TABLE training.model_backtest_zl AS
SELECT 
    model_name,
    horizon,
    anofox_fcst_ts_mae(actual, point_forecast) AS mae,
    anofox_fcst_ts_rmse(actual, point_forecast) AS rmse,
    anofox_fcst_ts_mape(actual, point_forecast) AS mape,
    anofox_fcst_ts_bias(actual, point_forecast) AS bias,
    anofox_fcst_ts_coverage(actual, lower_95, upper_95) AS coverage_95
FROM forecast.zl_anofox_panel
JOIN staging.market_daily ON ...
GROUP BY model_name, horizon;
```

**Baseline established. Neural models measured against this.**

---

## Phase 4: Feature Validation & Pruning (Days 10-11)

### 4.1 AnoFox Feature Importance
```sql
CREATE TABLE features.feature_importance_log AS
SELECT 
    feature_name,
    anofox_correlation(feature_value, zl_returns, window := 252) AS correlation_1y,
    anofox_statistics_ols(zl_returns, [feature_name]) AS ols_coefficient
FROM training.daily_ml_matrix;
```

### 4.2 Pruning Decision
- Keep features with:
  - |correlation| > 0.1 OR
  - Domain importance (manual) OR
  - AnoFox importance rank ≤ 100

**Target:** 80-120 production features

**Document:** `docs/reference/feature_catalog.md`

---

## Phase 5: Neural Models (Days 12-15)

### 5.1 Build Missing Neural Models

**Already exist:**
1. ✅ LSTM (`src/training/dl_round1/lstm_zl_minimal.py`)
2. ✅ TFT (`src/training/dl_round2/tft_zl_minimal.py`)

**To build:**
3. GRU (2-layer, 64-128 units)
4. TCN (kernel_size 3-5)
5. GARCH-LSTM hybrid (GARCH variance → LSTM)
6. HAR-LSTM hybrid (realized vol → LSTM)
7. N-BEATS (interpretable trend+seasonal blocks)

### 5.2 Training Protocol
- **Data:** AnoFox-cleaned from `training.daily_ml_matrix`
- **Validation:** Walk-forward (not k-fold)
- **Device:** Mac M4 MPS backend
- **Precision:** FP16 mixed precision (16GB RAM constraint)
- **Batch sizes:** LSTM/GRU ≤32, TCN ≤32, TFT/Attention ≤16

### 5.3 Neural Output
Write to MotherDuck: `forecast.zl_neural_panel`
- Same schema as `forecast.zl_anofox_panel`
- model_name identifies neural architecture

### 5.4 Beat-the-Baseline Threshold
**Neural model must beat AnoFox baseline by ≥5% MAPE or it's not used in ensemble**

---

## Phase 6: Meta-Ensemble (Days 16-18)

### 6.1 Build 7 Meta Models

**Stackers (XGBoost, LightGBM, Random Forest):**
- Input: 38 model forecasts (31 AnoFox + 7 neural) + regime features
- Target: Realized ZL outcomes
- Train on OOF (out-of-fold) predictions

**Residual Hybrids (ARIMA-LSTM, ETS-LSTM, MSTL-LSTM):**
- ARIMA/ETS/MSTL baseline (from AnoFox)
- LSTM trained on residuals
- Combined forecast = baseline + LSTM(residuals)

**Regime Model (Transformer-GARCH):**
- Input: GARCH variance + realized vol + macro stress
- Output: Regime classification (CALM/STRESSED/CRISIS)
- Purpose: Gate ensemble weights

### 6.2 Ensemble Selection
**Per horizon, select 5-7 models from 45 total**

**Criteria:**
- Lowest MAPE on validation
- Best coverage calibration (90%/95% intervals)
- Robustness across regimes

**Composition requirements:**
- ≥1 AnoFox statistical model
- ≥1 neural model (for horizons ≥1M)
- GARCH mandatory in CRISIS regime
- No single model > 0.40 weight

### 6.3 Regime-Aware Weights

**Regime Detection (from Transformer-GARCH):**
- **CALM:** Vol Z-score < 1.0
- **STRESSED:** Vol Z-score 1.0-2.0
- **CRISIS:** Vol Z-score > 2.0

**Weight Adjustments (See `/docs/architecture/zl_forecasting_engine_spec.md`)**

---

## Phase 7: Driver Group Attribution (Days 19-20)

### 7.1 Create Driver Groups
```sql
CREATE TABLE reference.driver_group (
    driver_group_id VARCHAR PRIMARY KEY,
    driver_group_name VARCHAR,
    description TEXT
);

INSERT INTO reference.driver_group VALUES
('BIOFUEL_ENERGY', 'Biofuel & Energy', 'RINs, EPA mandates, biodiesel demand, crude correlation'),
('TRADE_TARIFF', 'Trade & Tariffs', 'Trump, China, Brazil, Argentina trade relations'),
('WEATHER_SUPPLY', 'Weather & Supply', 'Drought, La Niña, harvest updates, USDA reports'),
('PALM_SUBSTITUTION', 'Palm Substitution', 'Indonesia levy, Malaysia stocks, palm spread'),
('MACRO_RISK', 'Macro Risk', 'VIX, DXY, Fed Funds, Treasury yields'),
('POSITIONING', 'Positioning', 'CFTC COT, managed money flows'),
('POLICY_REGULATION', 'Policy & Regulation', 'EPA, USDA, lobbying, executive orders'),
('TECHNICAL_REGIME', 'Technical & Regime', 'Volatility regime, technical indicators');
```

### 7.2 Map Features to Driver Groups
```sql
CREATE TABLE reference.feature_to_driver_group_map (
    feature_name VARCHAR PRIMARY KEY,
    driver_group_id VARCHAR REFERENCES reference.driver_group(driver_group_id)
);

-- Map all 80-120 production features
INSERT INTO reference.feature_to_driver_group_map VALUES
('eia_rin_price_d4', 'BIOFUEL_ENERGY'),
('policy_trump_score', 'TRADE_TARIFF'),
('weather_argentina_drought_zscore', 'WEATHER_SUPPLY'),
('fred_vix', 'MACRO_RISK'),
('cftc_managed_money_netlong', 'POSITIONING'),
-- ... all features mapped
;
```

### 7.3 Attribution Pipeline

**Three sources:**
1. **AnoFox Statistics:** OLS/Ridge regression per driver group
2. **TFT Attention:** Per-feature attention weights
3. **XGBoost SHAP:** Tree-based attribution

**Aggregate to driver groups:**
```sql
CREATE TABLE signals.driver_group_contribution AS
SELECT 
    date,
    horizon,
    driver_group_id,
    SUM(shap_value) AS total_contribution,
    AVG(ols_beta) AS avg_beta,
    AVG(tft_attention) AS avg_attention
FROM (
    -- SHAP values
    SELECT ... FROM xgboost_shap_output
    UNION ALL
    -- TFT attention
    SELECT ... FROM tft_attention_output
    UNION ALL
    -- OLS regression
    SELECT ... FROM anofox_statistics_ols_output
) 
JOIN reference.feature_to_driver_group_map USING (feature_name)
GROUP BY date, horizon, driver_group_id;
```

---

## Phase 8: Dashboard Integration (Days 21-22)

### 8.1 Quant Pages (PIN-Protected)

**`/quant-admin` - Training Results (Your Eyes Only)**

**PIN Implementation:**
- 4-digit code stored in env: `QUANT_ADMIN_PIN=1234`
- Middleware: `Dashboard/app/quant-admin/middleware.ts`
- Session cookie (24-hour expiry)

**UI Components:**
1. Model Leaderboard (45 models ranked by MAPE)
2. Training History Timeline
3. Feature Importance Heatmap
4. Ensemble Weights Dashboard
5. Model Drift Monitor

**API Route:** `/api/quant-admin` (POST with PIN, returns session token)

**Data Source:** `training.zl_model_runs`, `reference.model_registry`, `features.feature_importance_log`

---

**`/quant-reports` - AnoFox Reports (Chris-Visible)**

**UI Components:**
1. 7 Driver Group Performance Cards
2. Data Quality Gates (Pass/Fail status)
3. Forecast Accuracy Timeline
4. Validation Summary
5. Model Comparison Table

**API Route:** `/api/quant-reports`

**Data Source:** `ops.data_quality_zl_daily`, `forecast.zl_ensemble_output`, `training.model_backtest_zl`

### 8.2 Main Dashboard Updates

**Replace all BigQuery API calls with MotherDuck:**

**Before:**
```typescript
const client = new BigQuery();
const [rows] = await client.query('SELECT * FROM predictions.vw_zl_latest');
```

**After:**
```typescript
const conn = duckdb.connect(`md:usoil_intelligence?motherduck_token=${process.env.MOTHERDUCK_TOKEN}`);
const rows = await conn.all('SELECT * FROM forecast.zl_ensemble_output WHERE horizon = \'1M\' ORDER BY as_of_date DESC LIMIT 1');
```

### 8.3 Vercel Cron Jobs

**Daily (5:00 PM):** `/api/cron/daily-forecast`
1. Ingest latest ZL price (Databento)
2. Run AnoFox "big run" (31 models)
3. Run neural models (7 models)
4. Run meta-ensemble (7 models)
5. Update `forecast.zl_ensemble_output`
6. Log to `ops.pipeline_test_runs`

**15-Minute:** `/api/cron/bucket-scores`
1. Update driver group scores
2. Update `signals.big_eight_live`
3. Update `signals.driver_group_score_daily`

---

## Phase 9: Validation & Parallel Run (Days 23-24)

### 9.1 Walk-Forward Backtest
- Period: 2024-01-01 to 2025-12-03
- Method: Expanding window (retrain monthly)
- Metrics: MAPE, MAE, directional accuracy, Sharpe ratio

### 9.2 Parallel Run (1 Week)
- Run both BigQuery and MotherDuck forecasts
- Compare daily outputs
- Log discrepancies to `ops.migration_validation`

### 9.3 BigQuery Decommission
- Archive all BigQuery data to GCS (cold storage)
- Update `MASTER_PLAN.md` with MotherDuck architecture
- Remove BigQuery client libraries from codebase
- Delete BigQuery datasets (after 30-day archive verification)

---

## Phase 10: Documentation & CI/CD (Day 25)

### 10.1 Architecture Docs (Desk Manual Style)
- ✅ `docs/architecture/motherduck_platform.md`
- ✅ `docs/architecture/anofox_model_stack.md`
- ✅ `docs/architecture/bucket_system.md`
- ✅ `docs/architecture/zl_forecasting_engine_spec.md`

### 10.2 Runbooks
- ✅ `docs/runbooks/daily_pipeline.md`
- ✅ `docs/runbooks/model_retraining.md`
- ✅ `docs/runbooks/data_quality_gates.md`

### 10.3 Reference Docs
- ✅ `docs/reference/feature_catalog.md`
- ✅ `docs/reference/model_registry.md`
- ✅ `docs/reference/api_endpoints.md`
- ✅ `docs/reference/bucket_definitions.md`

### 10.4 CI/CD Tests
- MotherDuck connection test
- AnoFox extension load test
- Forecast validation test (MAPE threshold)
- Data quality gate test
- Ensemble weight sum = 1.0 test

---

## Success Criteria

- [ ] All 46 BigQuery tables migrated to MotherDuck
- [ ] AnoFox "big run" generates 31 baseline forecasts per horizon
- [ ] 7 neural models trained, beat baseline by ≥5% MAPE
- [ ] 7 meta-ensemble models trained
- [ ] 5-7 model ensemble selected per horizon
- [ ] Regime-aware weights implemented
- [ ] Driver group attribution operational
- [ ] PIN-protected `/quant-admin` deployed
- [ ] `/quant-reports` showing AnoFox QC results
- [ ] All 5 dashboard pages connected to MotherDuck
- [ ] Daily pipeline automated (Vercel Cron)
- [ ] BigQuery fully decommissioned

---

## Key Files

| File | Purpose |
|------|---------|
| `Scripts/migration/bq_to_motherduck.py` | BigQuery → Parquet → MotherDuck |
| `Data/db/sql/anofox_big_run.sql` | The "big run" - 31 models SQL |
| `src/training/dl_round1/gru_zl.py` | GRU neural model |
| `src/training/dl_round1/tcn_zl.py` | TCN neural model |
| `src/training/dl_round2/garch_lstm_hybrid.py` | GARCH-LSTM volatility |
| `src/training/dl_round2/har_lstm_hybrid.py` | HAR-LSTM multi-horizon vol |
| `src/training/dl_round2/nbeats_zl.py` | N-BEATS decomposition |
| `src/training/ensemble/xgboost_meta.py` | XGBoost stacker |
| `src/training/ensemble/lightgbm_regime_meta.py` | LightGBM regime-aware |
| `src/training/ensemble/arima_lstm_hybrid.py` | ARIMA-LSTM residual |
| `src/training/ensemble/ets_lstm_hybrid.py` | ETS-LSTM residual |
| `src/training/ensemble/mstl_lstm_hybrid.py` | MSTL-LSTM residual |
| `src/training/ensemble/transformer_garch.py` | Transformer-GARCH regime |
| `Dashboard/app/quant-admin/middleware.ts` | PIN protection |
| `Dashboard/app/api/motherduck/` | All MotherDuck API routes |
| `config/motherduck/ensemble_weights.yaml` | Production ensemble config |

---

**PLAN LOCKED.**

