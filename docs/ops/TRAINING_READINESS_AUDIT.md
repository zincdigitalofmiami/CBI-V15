# Training Readiness Audit

**Date:** December 16, 2025  
**Purpose:** Assess data completeness and requirements for test training run  
**Status:** 🟡 PARTIALLY READY - Critical gaps identified

---

## Executive Summary

**Can we train now?** ⚠️ **YES, but with limitations**

- ✅ **Core data exists**: 4,017 days of ZL price data (2010-2025)
- ✅ **Schemas deployed**: All 13 schemas exist in MotherDuck
- ✅ **Feature tables defined**: 93+ features mapped in SQL macros
- ❌ **Feature tables EMPTY**: No computed features yet
- ❌ **Recent data gaps**: Missing last 4 days (Dec 13-16, 2025)
- ⚠️ **EPA RIN data incomplete**: Only 3 weeks of history (should be 15+ years)

**Recommendation:** Run feature engineering pipeline first, then proceed with limited training run using available data.

---

## 1. RAW DATA INVENTORY

### ✅ GOOD: Core Market Data (Databento)

```
Table: raw.databento_futures_ohlcv_1d
- Total rows: 218,941
- Date range: 2010-06-07 to 2025-12-14
- Unique symbols: 56 (covers all 33 required symbols)
- ZL specific: 4,017 rows (15+ years of history)
```

**ZL Price Statistics:**
- Average: $43.89
- Range: $25.09 - $86.55
- Coverage: 2010-06-07 to 2025-12-12

**Data Quality:**
- ✅ No major gaps in historical data
- ⚠️ Missing last 4 days (weekend + Monday/Tuesday)
- ✅ Sufficient for training (3,500+ trading days)

### ✅ GOOD: Macro Data (FRED)

```
Table: raw.fred_economic
- Total rows: 252,655
- Date range: 2000-01-01 to 2025-12-15
- Unique series: 58 indicators
```

**Coverage includes:**
- Fed funds rate (DFEDTARU)
- Treasury yields (T10Y2Y curve)
- VIX (VIXCLS)
- Dollar Index (DXY)
- FX rates (BRL, CNY, MXN)
- Financial stress indices (NFCI, STLFSI4)

### ⚠️ PARTIAL: EPA RIN Prices (CRITICAL GAP)

```
Table: raw.epa_rin_prices
- Total rows: 208
- Date range: 2024-12-23 to 2025-12-15
- Unique RIN types: 4 (D3, D4, D5, D6)
```

**ISSUE:** Only 3 weeks of data, should have 15+ years (2010-present)

**Impact:**
- Biofuel bucket features will be incomplete
- Cannot compute historical RIN price trends
- Missing critical biodiesel demand signals

**Action Required:**
- Backfill EPA RIN prices from 2010-07-01 to 2024-12-22
- Source: https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information
- Priority: HIGH (blocks Biofuel bucket specialist training)

### ✅ GOOD: CFTC COT Data

```
Table: raw.cftc_cot
- Total rows: 4,506
- Date range: 2020-01-07 to 2025-11-25
- Unique symbols: 16
```

**Coverage:**
- 5+ years of positioning data
- Includes ZL, ZS, ZM, CL, HO, HG
- Weekly frequency (as expected)

### ✅ GOOD: USDA Data

```
Table: raw.usda_export_sales
- Total rows: 6,412
- Date range: 2020-01-02 to 2025-12-11
- Commodities: 3 (Soybeans, Corn, Wheat)

Table: raw.usda_wasde
- Total rows: 4,320
- Date range: 2020-01-12 to 2025-12-12
- Commodities: 3
```

**China Export Sales:**
- ✅ Critical for China bucket
- ✅ Weekly updates
- ✅ 5+ years of history

### ❌ MISSING: News/Sentiment Data

```
Table: raw.scrapecreators_news_buckets
- Total rows: 16 (TEST DATA ONLY)

Table: raw.bucket_news
- Total rows: 0 (EMPTY)

Table: raw.scrapecreators_trump
- Total rows: 0 (EMPTY)
```

**Impact:**
- Tariff bucket will lack sentiment signals
- Cannot compute Trump policy risk scores
- Missing Farm Policy News integration

**Action Required:**
- Deploy `trigger/ScrapeCreators/Scripts/collect_scrapecreators_news.py`
- Set up Farm Policy News scraper (farmpolicynews.illinois.edu)
- Priority: MEDIUM (can train without, but reduces accuracy)

---

## 2. FEATURE ENGINEERING STATUS

### ❌ CRITICAL: All Feature Tables Empty

```sql
-- ALL ZERO ROWS:
features.bucket_biofuel          -- 0 rows
features.bucket_china            -- 0 rows
features.bucket_crush            -- 0 rows
features.bucket_energy           -- 0 rows
features.bucket_fed              -- 0 rows
features.bucket_fx               -- 0 rows
features.bucket_tariff           -- 0 rows
features.bucket_volatility       -- 0 rows
features.bucket_scores           -- 0 rows
features.daily_ml_matrix_zl      -- 0 rows (MASTER TABLE)
features.technical_indicators_all_symbols  -- 0 rows
features.rolling_corr_beta       -- 0 rows
features.targets                 -- 0 rows
```

**Root Cause:** Feature engineering pipeline has never been run

**Action Required:**
1. Run `python src/engines/anofox/build_all_features.py`
2. This will execute SQL macros to populate all feature tables
3. Expected output: ~4,000 rows in `daily_ml_matrix_zl` (one per trading day)

### ✅ SQL Macros Defined

```
database/macros/
├── features.sql                              # Price/return macros
├── technical_indicators_all_symbols.sql      # 40+ technical indicators
├── cross_asset_features.sql                  # Correlations & spreads
├── big8_bucket_features.sql                  # Big 8 bucket scores
├── master_feature_matrix.sql                 # Final ML matrix (93+ features)
├── big8_cot_enhancements.sql                 # CFTC positioning
├── utils.sql                                 # Helper functions
├── asof_joins.sql                            # Time-series joins
└── anofox_guards.sql                         # Data quality checks
```

**Feature Count by Category:**
- Technical indicators: 40+ (RSI, MACD, Bollinger, ATR, Stochastic, etc.)
- Cross-asset correlations: 11 (ZL vs ZS/ZM/CL/HO/HG/DX)
- Fundamental spreads: 6 (Board crush, BOHO, crack spread, China pulse)
- Big 8 bucket scores: 16 (8 scores + 8 key metrics)
- Targets: 8 (4 horizons × 2 types: price + return)

**Total: 93+ features** (exact count: 81 base + targets)

---

## 3. TRAINING INFRASTRUCTURE

### ✅ AutoGluon Training Scripts Exist

```
src/training/autogluon/
├── mitra_trainer.py              # Mitra foundation model (Mac M4 Metal)
├── timeseries_trainer.py         # TimeSeriesPredictor wrapper
└── __init__.py

src/training/baselines/
├── lightgbm_zl.py                # LightGBM baseline
├── xgboost_zl.py                 # XGBoost baseline
└── catboost_zl.py                # CatBoost baseline
```

### ⚠️ Missing: Orchestration Scripts

**What we need to create:**

1. **Bucket Specialist Trainer** (`src/training/autogluon/train_bucket_specialists.py`)
   - Trains 8 TabularPredictor models (one per Big 8 bucket)
   - Uses `presets='extreme_quality'` (includes TabPFNv2, Mitra, TabICL)
   - Outputs OOF predictions to `training.bucket_predictions`

2. **Core ZL Trainer** (`src/training/autogluon/train_core_zl.py`)
   - Trains main TabularPredictor on all 93+ features
   - Problem type: `quantile` (P10/P50/P90)
   - Outputs to `training.core_ts_predictions`

3. **Meta Model Trainer** (`src/training/autogluon/train_meta_model.py`)
   - Fuses 9 specialist predictions (8 buckets + 1 core)
   - Learns optimal ensemble weights
   - Outputs final forecasts to `forecasts.zl_predictions`

4. **Full Pipeline Script** (`src/training/run_full_training.py`)
   - Orchestrates all 3 training stages
   - Handles data sync (MotherDuck → Local DuckDB)
   - Uploads predictions back to MotherDuck

### ✅ Local DuckDB Setup

**Path:** `data/duckdb/cbi_v15.duckdb`

**Status:** File deleted (needs recreation)

**Action Required:**
```bash
# Sync MotherDuck → Local DuckDB
python scripts/sync_motherduck_to_local.py

# This creates local mirror for fast training I/O
```

---

## 4. MINIMUM VIABLE TRAINING RUN

### What We Can Do RIGHT NOW

**Option A: Baseline Model (Simplest)**

Train a single LightGBM model on raw price data only:

```bash
# 1. Build minimal features (just ZL price + returns)
python -c "
import duckdb
import os

con = duckdb.connect(f'md:cbi_v15?motherduck_token={os.getenv(\"MOTHERDUCK_TOKEN\")}')

# Populate daily_ml_matrix_zl with just price/returns
con.execute('''
    INSERT OR REPLACE INTO features.daily_ml_matrix_zl
    SELECT 
        as_of_date,
        'ZL' as symbol,
        close,
        LAG(close, 1) OVER (ORDER BY as_of_date) as lag_close_1d,
        LAG(close, 5) OVER (ORDER BY as_of_date) as lag_close_5d,
        LAG(close, 21) OVER (ORDER BY as_of_date) as lag_close_21d,
        LN(close / LAG(close, 1) OVER (ORDER BY as_of_date)) as log_ret_1d,
        LN(close / LAG(close, 5) OVER (ORDER BY as_of_date)) as log_ret_5d,
        LN(close / LAG(close, 21) OVER (ORDER BY as_of_date)) as log_ret_21d,
        LEAD(close, 5) OVER (ORDER BY as_of_date) as target_price_1w,
        LEAD(close, 21) OVER (ORDER BY as_of_date) as target_price_1m
    FROM raw.databento_futures_ohlcv_1d
    WHERE symbol = 'ZL'
    ORDER BY as_of_date
''')

con.close()
"

# 2. Sync to local
python scripts/sync_motherduck_to_local.py

# 3. Train baseline
python src/training/baselines/lightgbm_zl.py
```

**Expected Output:**
- Model artifact: `data/models/lightgbm_zl_baseline.pkl`
- Training metrics: MAPE, RMSE, directional accuracy
- Validation predictions: 1-week and 1-month horizons

**Limitations:**
- No Big 8 bucket features
- No cross-asset correlations
- No sentiment signals
- Single model (no ensemble)

### Option B: Full Feature Pipeline (Recommended)

Run complete feature engineering, then train:

```bash
# 1. Build ALL features (93+)
python src/engines/anofox/build_all_features.py

# Expected output:
# - features.technical_indicators_all_symbols: ~132,000 rows (33 symbols × 4,000 days)
# - features.bucket_scores: ~4,000 rows
# - features.daily_ml_matrix_zl: ~4,000 rows × 93 features

# 2. Sync to local
python scripts/sync_motherduck_to_local.py

# 3. Train AutoGluon TabularPredictor
python src/training/autogluon/train_core_zl.py
```

**Expected Output:**
- 10-15 models trained (LightGBM, CatBoost, XGBoost, Neural Nets)
- Automatic ensemble (WeightedEnsemble_L2)
- Quantile predictions (P10/P50/P90)
- Model artifacts: `data/models/ag_zl_core/`

**Limitations:**
- EPA RIN prices incomplete (Biofuel bucket weak)
- No news sentiment (Tariff bucket weak)
- Still single-stage (no bucket specialists yet)

---

## 5. CRITICAL BLOCKERS FOR FULL V15.1 TRAINING

### 🔴 BLOCKER 1: EPA RIN Price Backfill

**What's Missing:**
- Historical RIN prices from 2010-07-01 to 2024-12-22
- ~750 weeks of D3/D4/D5/D6 prices

**Why Critical:**
- Biofuel bucket specialist cannot train without RIN history
- RIN prices are PRIMARY driver for ZL (soybean oil → biodiesel)
- Scott Irwin's models show 75% R² between RIN prices and ZL

**Action:**
```python
# Create: trigger/EIA_EPA/Scripts/backfill_epa_rin_prices.py
# - Scrape EPA EMTS historical data
# - Weekly volume-weighted averages
# - Insert into raw.epa_rin_prices
```

**Timeline:** 2-4 hours (manual scraping + data cleaning)

### 🟡 BLOCKER 2: Feature Engineering Pipeline

**What's Missing:**
- `build_all_features.py` has never been run
- All feature tables empty

**Why Critical:**
- Cannot train without features
- Need 93+ features for full model

**Action:**
```bash
python src/engines/anofox/build_all_features.py
```

**Timeline:** 5-10 minutes (SQL execution)

### 🟡 BLOCKER 3: Training Orchestration Scripts

**What's Missing:**
- Bucket specialist trainer
- Meta model trainer
- Full pipeline orchestrator

**Why Critical:**
- V15.1 architecture requires 3-stage training
- Cannot run full ensemble without orchestration

**Action:**
- Create 4 new training scripts (see Section 3)

**Timeline:** 4-6 hours (development + testing)

### 🟢 OPTIONAL: News Sentiment Integration

**What's Missing:**
- Farm Policy News scraper
- ScrapeCreators Trump posts
- News bucket aggregation

**Why Optional:**
- Can train without sentiment
- Reduces Tariff bucket accuracy but not critical
- Other buckets provide sufficient signal

**Action:**
- Deploy news scrapers
- Backfill 2020-present

**Timeline:** 8-12 hours (scraper development + backfill)

---

## 6. RECOMMENDED ACTION PLAN

### Phase 1: Quick Test Run (TODAY)

**Goal:** Validate training infrastructure with minimal features

```bash
# 1. Build minimal features (10 minutes)
python src/engines/anofox/build_all_features.py

# 2. Sync to local (5 minutes)
python scripts/sync_motherduck_to_local.py

# 3. Train baseline (30 minutes)
python src/training/baselines/lightgbm_zl.py
```

**Success Criteria:**
- ✅ Feature tables populated
- ✅ Model trains without errors
- ✅ Predictions generated for 1w/1m horizons
- ✅ MAPE < 10% on validation set

### Phase 2: EPA RIN Backfill (TOMORROW)

**Goal:** Complete Biofuel bucket data

```bash
# 1. Create backfill script
# 2. Scrape EPA historical data
# 3. Insert into raw.epa_rin_prices
# 4. Rebuild features
```

**Success Criteria:**
- ✅ 750+ weeks of RIN prices (2010-2025)
- ✅ D3/D4/D5/D6 coverage complete
- ✅ Biofuel bucket features populated

### Phase 3: Full V15.1 Training (WEEK 1)

**Goal:** Deploy complete 3-stage ensemble

```bash
# 1. Create orchestration scripts
# 2. Train 8 bucket specialists
# 3. Train core ZL predictor
# 4. Train meta model
# 5. Generate forecasts
```

**Success Criteria:**
- ✅ 9 specialist models trained
- ✅ Meta model fuses predictions
- ✅ Forecasts uploaded to MotherDuck
- ✅ Dashboard displays predictions

---

## 7. DATA QUALITY CHECKS

### Before Training, Verify:

```sql
-- 1. Feature table row counts
SELECT 
    'daily_ml_matrix_zl' as table_name,
    COUNT(*) as row_count,
    MIN(as_of_date) as min_date,
    MAX(as_of_date) as max_date
FROM features.daily_ml_matrix_zl;

-- Expected: ~4,000 rows, 2010-2025

-- 2. No NULL targets
SELECT COUNT(*) 
FROM features.daily_ml_matrix_zl
WHERE target_price_1w IS NULL OR target_price_1m IS NULL;

-- Expected: ~21 rows (last month has no future data)

-- 3. Feature completeness
SELECT 
    COUNT(*) as total_rows,
    COUNT(close) as has_close,
    COUNT(log_ret_1d) as has_returns,
    COUNT(sma_20) as has_sma,
    COUNT(rsi_14) as has_rsi,
    COUNT(board_crush_spread) as has_crush,
    COUNT(vix) as has_vix
FROM features.daily_ml_matrix_zl;

-- All counts should equal total_rows

-- 4. No extreme outliers
SELECT 
    MIN(close) as min_price,
    MAX(close) as max_price,
    AVG(close) as avg_price,
    STDDEV(close) as std_price
FROM features.daily_ml_matrix_zl;

-- ZL range: $25-$87 (historical)
```

---

## 8. TRAINING CONFIGURATION

### Recommended Settings for Test Run

```python
# AutoGluon TabularPredictor
predictor = TabularPredictor(
    label='target_price_1w',
    problem_type='quantile',  # P10/P50/P90
    eval_metric='pinball_loss',
    quantile_levels=[0.1, 0.5, 0.9],
    path='data/models/ag_zl_test',
    verbosity=2
)

# Training
predictor.fit(
    train_data=train_df,
    presets='medium_quality',  # Start with medium (faster)
    time_limit=1800,  # 30 minutes
    num_bag_folds=5,  # 5-fold CV
    num_bag_sets=1,
    num_stack_levels=1,
    excluded_model_types=['KNN', 'XT'],  # Exclude slow models
)
```

### For Production (After Test)

```python
# Switch to extreme_quality for foundation models
predictor.fit(
    train_data=train_df,
    presets='extreme_quality',  # Includes TabPFNv2, Mitra, TabICL
    time_limit=7200,  # 2 hours
    num_bag_folds=10,
    num_bag_sets=2,
    num_stack_levels=2,
)
```

---

## 9. EXPECTED TRAINING TIME

### Test Run (Minimal Features)

- Feature engineering: 10 minutes
- Data sync: 5 minutes
- Model training: 30 minutes
- **Total: 45 minutes**

### Full Run (93+ Features, Single Model)

- Feature engineering: 15 minutes
- Data sync: 10 minutes
- Model training: 2 hours (extreme_quality)
- **Total: 2.5 hours**

### Full V15.1 Pipeline (3-Stage Ensemble)

- Feature engineering: 15 minutes
- Data sync: 10 minutes
- Bucket specialists (8 models): 4 hours
- Core ZL predictor: 2 hours
- Meta model: 30 minutes
- **Total: 7 hours**

---

## 10. SUMMARY & NEXT STEPS

### Current Status: 🟡 READY FOR LIMITED TEST

**What Works:**
- ✅ 15+ years of ZL price data
- ✅ 58 FRED macro indicators
- ✅ CFTC positioning data
- ✅ USDA export sales
- ✅ SQL macros defined (93+ features)
- ✅ Training scripts exist

**What's Missing:**
- ❌ Feature tables empty (need to run pipeline)
- ❌ EPA RIN prices incomplete (only 3 weeks)
- ❌ News sentiment data (optional)
- ❌ Orchestration scripts (for full ensemble)

### Immediate Action (Next 1 Hour)

```bash
# Step 1: Build features
python src/engines/anofox/build_all_features.py

# Step 2: Sync to local
python scripts/sync_motherduck_to_local.py

# Step 3: Train baseline
python src/training/baselines/lightgbm_zl.py
```

### This Week

1. **Day 1:** Complete test run (above)
2. **Day 2:** Backfill EPA RIN prices
3. **Day 3-4:** Create orchestration scripts
4. **Day 5:** Full V15.1 training run

### Success Metrics

**Test Run:**
- ✅ Model trains without errors
- ✅ MAPE < 10% on validation
- ✅ Predictions look reasonable

**Full Run:**
- ✅ All 9 specialists trained
- ✅ Ensemble beats individual models
- ✅ Forecasts uploaded to MotherDuck
- ✅ Dashboard displays predictions

---

## APPENDIX: File Checklist

### ✅ Exists
- `src/engines/anofox/build_all_features.py`
- `src/training/baselines/lightgbm_zl.py`
- `src/training/autogluon/mitra_trainer.py`
- `database/macros/*.sql` (9 files)
- `scripts/sync_motherduck_to_local.py`

### ❌ Needs Creation
- `src/training/autogluon/train_bucket_specialists.py`
- `src/training/autogluon/train_core_zl.py`
- `src/training/autogluon/train_meta_model.py`
- `src/training/run_full_training.py`
- `trigger/EIA_EPA/Scripts/backfill_epa_rin_prices.py`

---

**Last Updated:** December 16, 2025  
**Next Review:** After test training run completion
