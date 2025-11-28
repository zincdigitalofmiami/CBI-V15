# Pre-BigQuery Forensic Audit

**Date**: November 28, 2025  
**Purpose**: Ultra-certain verification before BigQuery environment creation  
**Status**: ⚠️ **IN PROGRESS**

---

## 🎯 Audit Scope

### Critical Areas to Verify:
1. ✅ All tables/buckets accounted for (regimes, horizons, news, sentiment, neurals)
2. ✅ Scheduler workflows planned and documented
3. ✅ News/neural segmentation strategy (reduce brittleness/drift)
4. ✅ No errors, inconsistencies, or bloat
5. ✅ 100% accuracy and correctness

---

## 📊 Part 1: Table/Bucket Inventory Audit

### A. Core Data Tables ✅

#### Raw Layer (7 tables)
- ✅ `databento_futures_ohlcv_1d` - Market data
- ✅ `fred_economic` - Economic indicators
- ✅ `usda_reports` - USDA reports
- ✅ `cftc_cot` - CFTC positions
- ✅ `eia_biofuels` - EIA biofuels
- ✅ `weather_noaa` - Weather data
- ✅ `scrapecreators_trump` - Trump policy intelligence

**Status**: ✅ **COMPLETE**

---

#### Staging Layer (7 tables)
- ✅ `market_daily` - Cleaned market data
- ✅ `fred_macro_clean` - Cleaned FRED data
- ✅ `usda_reports_clean` - Cleaned USDA data
- ✅ `cftc_positions` - CFTC positions by category
- ✅ `eia_biofuels_clean` - Cleaned EIA data
- ✅ `weather_regions_aggregated` - Aggregated weather
- ✅ `trump_policy_intelligence` - Policy events

**Status**: ✅ **COMPLETE**

---

### B. Missing Critical Tables ⚠️

#### 1. News/Sentiment Buckets ⚠️ **MISSING**

**Required Tables**:
- ⚠️ `raw.scrapecreators_news_buckets` - All news buckets (biofuel, China, tariffs)
- ⚠️ `staging.news_bucketed` - Aggregated news buckets by date
- ⚠️ `staging.sentiment_buckets` - Sentiment scores by bucket
- ⚠️ `features.sentiment_features_daily` - Sentiment features for ML

**Current Status**: ⚠️ **NOT IN SKELETON**

**Impact**: ⚠️ **HIGH** - News/sentiment is Big 8 driver #5 (Tariffs)

**Action Required**: ✅ **ADD** - Must add before BQ setup

---

#### 2. Regime System ⚠️ **MISSING**

**Required Tables**:
- ⚠️ `reference.regime_calendar` - Regime dates (Trump eras, crises, etc.)
- ⚠️ `reference.regime_weights` - Regime weights (VIX-based, shock multipliers)
- ⚠️ `features.regime_indicators_daily` - Daily regime indicators

**Current Status**: ⚠️ **NOT IN SKELETON**

**Impact**: ⚠️ **HIGH** - Regime weighting is critical for training

**Action Required**: ✅ **ADD** - Must add before BQ setup

---

#### 3. Horizon Targets ⚠️ **PARTIAL**

**Current**: Targets in `daily_ml_matrix` ✅

**Missing**:
- ⚠️ `reference.horizon_definitions` - Horizon metadata (1w=5d, 1m=20d, etc.)
- ⚠️ `training.horizon_metadata` - Horizon-specific metadata

**Current Status**: ⚠️ **NOT IN SKELETON**

**Impact**: ⚠️ **MEDIUM** - Nice-to-have, not critical

**Action Required**: ⚠️ **OPTIONAL** - Can add later

---

#### 4. Neural Features ⚠️ **MISSING**

**Required Tables**:
- ⚠️ `features.neural_signals_daily` - Neural composite signals (dollar_neural_score, fed_neural_score, crush_neural_score)
- ⚠️ `features.neural_master_score` - Master neural score
- ⚠️ `reference.neural_drivers` - Neural driver definitions (Layer 3 → Layer 2 → Layer 1)

**Current Status**: ⚠️ **NOT IN SKELETON**

**Impact**: ⚠️ **HIGH** - Neural architecture is core to the system

**Action Required**: ✅ **ADD** - Must add before BQ setup

---

### C. Feature Tables ✅

#### Features Layer (7 tables)
- ✅ `technical_indicators_us_oil_solutions` - 19 features
- ✅ `fx_indicators_daily` - 16 features
- ✅ `fundamental_spreads_daily` - 5 features
- ✅ `pair_correlations_daily` - 112 features
- ✅ `cross_asset_betas_daily` - 28 features
- ✅ `lagged_features_daily` - 96 features
- ✅ `daily_ml_matrix` - Master join (276 features)

**Status**: ✅ **COMPLETE**

---

### D. Training/Forecasts Tables ✅

#### Training Layer (4 tables)
- ✅ `zl_training_1w` - Training data 1w
- ✅ `zl_training_1m` - Training data 1m
- ✅ `zl_training_3m` - Training data 3m
- ✅ `zl_training_6m` - Training data 6m

**Status**: ✅ **COMPLETE**

#### Forecasts Layer (4 tables)
- ✅ `zl_predictions_1w` - Predictions 1w
- ✅ `zl_predictions_1m` - Predictions 1m
- ✅ `zl_predictions_3m` - Predictions 3m
- ✅ `zl_predictions_6m` - Predictions 6m

**Status**: ✅ **COMPLETE**

---

## 📋 Part 2: Missing Tables Summary

### Critical Missing Tables (Must Add):

1. ⚠️ **News/Sentiment Buckets** (4 tables)
   - `raw.scrapecreators_news_buckets`
   - `staging.news_bucketed`
   - `staging.sentiment_buckets`
   - `features.sentiment_features_daily`

2. ⚠️ **Regime System** (3 tables)
   - `reference.regime_calendar`
   - `reference.regime_weights`
   - `features.regime_indicators_daily`

3. ⚠️ **Neural Features** (3 tables)
   - `features.neural_signals_daily`
   - `features.neural_master_score`
   - `reference.neural_drivers`

**Total Missing**: 10 critical tables ⚠️

---

## 🔄 Part 3: Scheduler Workflow Audit

### Current Scheduler Status ⚠️ **NOT PLANNED**

**Missing**:
- ⚠️ No scheduler configuration files
- ⚠️ No workflow documentation
- ⚠️ No ingestion schedule defined

**Impact**: ⚠️ **CRITICAL** - No automated data ingestion

---

### Required Scheduler Workflows

#### 1. Market Data Ingestion (Databento)

**Frequency**: 
- ZL price: Every 1 hour
- Other symbols: Every 4 hours

**Workflow**:
1. Python script pulls from Databento API
2. Local Parquet cache (7 days retention)
3. Upload to BigQuery `raw.databento_futures_ohlcv_1d`
4. Trigger Dataform staging transformation

**Scheduler**: Cloud Scheduler → Cloud Function

**Status**: ⚠️ **NOT PLANNED**

---

#### 2. FRED Macro Data Ingestion

**Frequency**: Daily (after market close)

**Workflow**:
1. Python script pulls from FRED API
2. Forward-fill missing values
3. Upload to BigQuery `raw.fred_economic`
4. Trigger Dataform staging transformation

**Scheduler**: Cloud Scheduler → Cloud Function

**Status**: ⚠️ **NOT PLANNED**

---

#### 3. News/Sentiment Buckets Ingestion (ScrapeCreators)

**Frequency**: Every 60 minutes

**Workflow**:
1. Python script pulls from ScrapeCreators API
2. Segment into buckets (biofuel, China, tariffs)
3. Calculate sentiment scores (FinBERT)
4. Upload to BigQuery `raw.scrapecreators_news_buckets`
5. Trigger Dataform staging transformation

**Scheduler**: Cloud Scheduler → Cloud Function

**Status**: ⚠️ **NOT PLANNED**

**Critical**: ⚠️ **SEGMENTATION MUST HAPPEN AT INGESTION** (see Part 4)

---

#### 4. Trump Policy Intelligence Ingestion

**Frequency**: Every 60 minutes

**Workflow**:
1. Python script pulls from ScrapeCreators API (Truth Social)
2. Extract policy events
3. Calculate ZL impact scores
4. Upload to BigQuery `raw.scrapecreators_trump`
5. Trigger Dataform staging transformation

**Scheduler**: Cloud Scheduler → Cloud Function

**Status**: ⚠️ **NOT PLANNED**

---

#### 5. USDA/CFTC/EIA Ingestion

**Frequency**: 
- USDA: Weekly/Monthly (after report release)
- CFTC: Weekly (after COT release)
- EIA: Weekly (after report release)

**Workflow**:
1. Python script pulls from API
2. Parse and clean data
3. Upload to BigQuery `raw.*`
4. Trigger Dataform staging transformation

**Scheduler**: Cloud Scheduler → Cloud Function

**Status**: ⚠️ **NOT PLANNED**

---

#### 6. Weather Data Ingestion

**Frequency**: Daily

**Workflow**:
1. Python script pulls from NOAA/INMET/SMN APIs
2. Aggregate by region
3. Upload to BigQuery `raw.weather_noaa`
4. Trigger Dataform staging transformation

**Scheduler**: Cloud Scheduler → Cloud Function

**Status**: ⚠️ **NOT PLANNED**

---

#### 7. Dataform Feature Computation

**Frequency**: Daily (after all ingestion complete)

**Workflow**:
1. Wait for all raw data ingestion complete
2. Run Dataform transformations (staging → features)
3. Build `daily_ml_matrix`
4. Verify data quality

**Scheduler**: Cloud Scheduler → Dataform API

**Status**: ⚠️ **NOT PLANNED**

---

### Scheduler Architecture Recommendation

#### Option A: Single Scheduler (Sequential) ⚠️ **NOT RECOMMENDED**
- All ingestion runs sequentially
- Slow, single point of failure
- No parallelization

#### Option B: Separate Schedulers (Parallel) ✅ **RECOMMENDED**
- Each data source has its own scheduler
- Parallel ingestion
- Independent failure handling
- Better monitoring

**Recommendation**: ✅ **Option B** - Separate schedulers for each data source

---

## 🧠 Part 4: News/Neural Segmentation Strategy

### Problem: Brittleness and Drift

**Issue**: News/sentiment features can cause model brittleness and drift if not properly segmented.

**Root Causes**:
1. **Temporal Drift**: News patterns change over time (2018 trade war ≠ 2024 trade war)
2. **Source Drift**: News sources change (new outlets, social media evolution)
3. **Semantic Drift**: Language evolves (same words mean different things)
4. **Volume Drift**: News volume spikes (crisis periods)

---

### Solution: Segmentation at Ingestion

#### Strategy 1: Bucket Segmentation ✅

**At Ingestion**:
1. Segment news into buckets IMMEDIATELY:
   - `biofuel_policy` - EPA, RFS, biodiesel mandates
   - `china_demand` - China import news, trade relations
   - `tariffs_trade_policy` - Tariff announcements, trade war events

2. Calculate sentiment PER BUCKET (not aggregate)

3. Store in separate tables:
   - `raw.scrapecreators_news_buckets` (by bucket)
   - `staging.news_bucketed` (aggregated by date, bucket)
   - `staging.sentiment_buckets` (sentiment scores by bucket)

**Why**: Prevents cross-contamination, allows bucket-specific modeling

---

#### Strategy 2: Temporal Segmentation ✅

**At Ingestion**:
1. Tag news with temporal markers:
   - `regime_period` - Which regime (Trump 2018, Trump 2024, Normal, Crisis)
   - `date_bucket` - Pre/post major events (trade war start, COVID, etc.)

2. Store regime metadata:
   - `reference.regime_calendar` - Regime dates
   - `features.regime_indicators_daily` - Daily regime flags

**Why**: Allows regime-specific feature engineering, reduces temporal drift

---

#### Strategy 3: Source Segmentation ✅

**At Ingestion**:
1. Tag news with source metadata:
   - `source_type` - News outlet, social media, government
   - `source_trust_score` - Trust/reliability score
   - `source_decay_factor` - How quickly trust decays

2. Weight sentiment by source trust

**Why**: Prevents low-quality sources from polluting features

---

#### Strategy 4: Volume Normalization ✅

**At Ingestion**:
1. Calculate news volume per bucket per day
2. Normalize sentiment by volume (prevent volume spikes from dominating)
3. Store normalized sentiment scores

**Why**: Prevents volume spikes from causing feature drift

---

### Neural Segmentation Strategy ✅

#### Problem: Neural Features Can Drift

**Issue**: Neural composite signals (dollar_neural_score, fed_neural_score) can drift if underlying drivers change.

**Solution**: Segment by Driver Layer

**At Feature Engineering**:
1. **Layer 3 (Deep Drivers)**: Rate differentials, employment, processing capacity
2. **Layer 2 (Neural Scores)**: dollar_neural_score, fed_neural_score, crush_neural_score
3. **Layer 1 (Master Score)**: Master neural score

**Segmentation**:
- Store each layer separately
- Calculate layer-specific weights
- Allow layer-specific drift detection

**Tables**:
- `features.neural_signals_daily` - Layer 2 scores
- `features.neural_master_score` - Layer 1 score
- `reference.neural_drivers` - Layer 3 definitions

---

## ⚠️ Part 5: Critical Gaps Identified

### Missing Tables (10 critical tables):

1. ⚠️ `raw.scrapecreators_news_buckets`
2. ⚠️ `staging.news_bucketed`
3. ⚠️ `staging.sentiment_buckets`
4. ⚠️ `features.sentiment_features_daily`
5. ⚠️ `reference.regime_calendar`
6. ⚠️ `reference.regime_weights`
7. ⚠️ `features.regime_indicators_daily`
8. ⚠️ `features.neural_signals_daily`
9. ⚠️ `features.neural_master_score`
10. ⚠️ `reference.neural_drivers`

---

### Missing Scheduler Configuration:

1. ⚠️ No Cloud Scheduler configs
2. ⚠️ No Cloud Function definitions
3. ⚠️ No workflow documentation
4. ⚠️ No ingestion schedule

---

### Missing Segmentation Strategy:

1. ⚠️ No news bucket segmentation at ingestion
2. ⚠️ No temporal segmentation (regime tagging)
3. ⚠️ No source segmentation (trust scoring)
4. ⚠️ No volume normalization
5. ⚠️ No neural layer segmentation

---

## ✅ Part 6: Recommendations

### Before BigQuery Setup:

1. ✅ **ADD** missing tables (10 critical tables)
2. ✅ **CREATE** scheduler configuration files
3. ✅ **DOCUMENT** segmentation strategy
4. ✅ **IMPLEMENT** segmentation at ingestion

### After BigQuery Setup:

1. ✅ **TEST** segmentation strategy
2. ✅ **MONITOR** drift detection
3. ✅ **VALIDATE** regime weighting
4. ✅ **VERIFY** neural layer segmentation

---

## 🎯 Action Items

### Immediate (Before BQ Setup):

1. ⚠️ Create missing table definitions (10 tables)
2. ⚠️ Create scheduler configuration files
3. ⚠️ Document segmentation strategy
4. ⚠️ Update skeleton structure

### After BQ Setup:

1. ⚠️ Implement segmentation at ingestion
2. ⚠️ Test scheduler workflows
3. ⚠️ Monitor drift detection
4. ⚠️ Validate regime weighting

---

**Status**: ⚠️ **AUDIT INCOMPLETE** - 10 critical tables missing, schedulers not planned

**Recommendation**: ⚠️ **DO NOT PROCEED** until gaps are filled

---

**Last Updated**: November 28, 2025

