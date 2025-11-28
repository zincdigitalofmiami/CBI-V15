# Scheduler Workflow Architecture

**Date**: November 28, 2025  
**Status**: ✅ **PLANNED** - Separate schedulers for parallel ingestion

---

## 🎯 Architecture: Separate Schedulers (Parallel)

### Why Separate Schedulers?

**Benefits**:
- ✅ Parallel ingestion (faster)
- ✅ Independent failure handling
- ✅ Better monitoring (per-source metrics)
- ✅ Easier debugging (isolated failures)

**Drawbacks**:
- ⚠️ More complex (multiple schedulers to manage)
- ⚠️ Need coordination for Dataform triggers

**Verdict**: ✅ **SEPARATE SCHEDULERS** - Benefits outweigh complexity

---

## 📊 Scheduler Workflow

### Workflow Overview:

```
Cloud Scheduler (per data source)
    ↓
Cloud Function (ingestion script)
    ↓
BigQuery Raw Table
    ↓
Dataform Trigger (staging transformation)
    ↓
BigQuery Staging Table
    ↓
Dataform Trigger (feature computation)
    ↓
BigQuery Features Table
    ↓
Dataform Trigger (daily_ml_matrix build)
    ↓
BigQuery Training Table
```

---

## 🔄 Detailed Workflows

### 1. Market Data (Databento)

**Scheduler**: `databento-zl-price-hourly` (every 1 hour)

**Workflow**:
1. Cloud Scheduler triggers Cloud Function
2. Cloud Function runs `src/ingestion/databento/collect_daily.py`
3. Script pulls from Databento API (ZL price)
4. Local Parquet cache (7 days retention)
5. Upload to BigQuery `raw.databento_futures_ohlcv_1d`
6. Trigger Dataform staging transformation
7. Dataform builds `staging.market_daily`

**Dependencies**: None (independent)

**Failure Handling**: Retry 3x, max backoff 3600s

---

### 2. FRED Macro Data

**Scheduler**: `fred-macro-daily` (daily at 6 PM ET)

**Workflow**:
1. Cloud Scheduler triggers Cloud Function
2. Cloud Function runs `src/ingestion/fred/collect_comprehensive.py`
3. Script pulls from FRED API (55-60 series)
4. Forward-fill missing values
5. Upload to BigQuery `raw.fred_economic`
6. Trigger Dataform staging transformation
7. Dataform builds `staging.fred_macro_clean`

**Dependencies**: None (independent)

**Failure Handling**: Retry 3x, max backoff 1800s

---

### 3. News/Sentiment Buckets (ScrapeCreators)

**Scheduler**: `scrapecreators-news-buckets-hourly` (every 1 hour)

**Workflow**:
1. Cloud Scheduler triggers Cloud Function
2. Cloud Function runs `src/ingestion/scrapecreators/buckets/collect_news_buckets.py`
3. Script pulls from ScrapeCreators API
4. **SEGMENT INTO BUCKETS AT INGESTION** (biofuel, China, tariffs)
5. Calculate sentiment per bucket (FinBERT)
6. Tag with temporal markers (regime, date buckets)
7. Tag with source metadata (trust scores)
8. Normalize sentiment by volume
9. Upload to BigQuery `raw.scrapecreators_news_buckets`
10. Trigger Dataform staging transformation
11. Dataform builds `staging.news_bucketed` and `staging.sentiment_buckets`

**Dependencies**: None (independent)

**Failure Handling**: Retry 3x, max backoff 1800s

**Critical**: ✅ **SEGMENTATION MUST HAPPEN AT STEP 4** (before BigQuery)

---

### 4. Trump Policy Intelligence

**Scheduler**: `scrapecreators-trump-hourly` (every 1 hour)

**Workflow**:
1. Cloud Scheduler triggers Cloud Function
2. Cloud Function runs `src/ingestion/scrapecreators/collect_trump_posts.py`
3. Script pulls from ScrapeCreators API (Truth Social)
4. Extract policy events
5. Calculate ZL impact scores
6. Upload to BigQuery `raw.scrapecreators_trump`
7. Trigger Dataform staging transformation
8. Dataform builds `staging.trump_policy_intelligence`

**Dependencies**: None (independent)

**Failure Handling**: Retry 3x, max backoff 1800s

---

### 5. USDA/CFTC/EIA Data

**Schedulers**: 
- `usda-reports-weekly` (Monday 10 AM ET)
- `cftc-cot-weekly` (Friday 10 AM ET)
- `eia-biofuels-weekly` (Wednesday 10 AM ET)

**Workflow** (similar for all):
1. Cloud Scheduler triggers Cloud Function
2. Cloud Function runs ingestion script
3. Script pulls from API
4. Parse and clean data
5. Upload to BigQuery `raw.*`
6. Trigger Dataform staging transformation
7. Dataform builds `staging.*`

**Dependencies**: None (independent)

**Failure Handling**: Retry 3x, max backoff 3600s

---

### 6. Weather Data

**Scheduler**: `weather-noaa-daily` (daily at 2 AM ET)

**Workflow**:
1. Cloud Scheduler triggers Cloud Function
2. Cloud Function runs `src/ingestion/weather/collect_noaa_comprehensive.py`
3. Script pulls from NOAA/INMET/SMN APIs
4. Aggregate by region (US Midwest, Brazil, Argentina)
5. Upload to BigQuery `raw.weather_noaa`
6. Trigger Dataform staging transformation
7. Dataform builds `staging.weather_regions_aggregated`

**Dependencies**: None (independent)

**Failure Handling**: Retry 3x, max backoff 1800s

---

### 7. Dataform Feature Computation

**Scheduler**: `dataform-features-daily` (daily at 3 AM ET)

**Workflow**:
1. Cloud Scheduler triggers Dataform API
2. **WAIT FOR ALL INGESTION COMPLETE** (check completion flags)
3. Run Dataform transformations (staging → features)
4. Build feature tables:
   - `features.technical_indicators_us_oil_solutions`
   - `features.fx_indicators_daily`
   - `features.fundamental_spreads_daily`
   - `features.pair_correlations_daily`
   - `features.cross_asset_betas_daily`
   - `features.lagged_features_daily`
   - `features.sentiment_features_daily`
   - `features.regime_indicators_daily`
   - `features.neural_signals_daily`
5. Verify data quality

**Dependencies**: All ingestion schedulers (must wait for completion)

**Failure Handling**: Retry 2x, max backoff 3600s

---

### 8. Daily ML Matrix Build

**Scheduler**: `dataform-daily-ml-matrix-daily` (daily at 4 AM ET)

**Workflow**:
1. Cloud Scheduler triggers Dataform API
2. **WAIT FOR FEATURE COMPUTATION COMPLETE**
3. Build `features.daily_ml_matrix` (master join)
4. Verify data quality
5. Build training split views:
   - `training.daily_ml_matrix_train`
   - `training.daily_ml_matrix_val`
   - `training.daily_ml_matrix_test`

**Dependencies**: Feature computation scheduler (must wait for completion)

**Failure Handling**: Retry 2x, max backoff 3600s

---

## 🔄 Coordination Strategy

### Problem: How to coordinate schedulers?

**Solution**: Completion Flags in BigQuery

```sql
-- Create completion tracking table
CREATE TABLE `ops.ingestion_completion` (
  date DATE,
  source STRING,
  completed_at TIMESTAMP,
  status STRING
);

-- Each ingestion script updates completion flag
UPDATE `ops.ingestion_completion`
SET completed_at = CURRENT_TIMESTAMP(), status = 'completed'
WHERE date = CURRENT_DATE() AND source = 'databento_zl';
```

**Dataform scheduler checks**:
```sql
-- Wait for all ingestion complete
SELECT COUNT(*) as pending
FROM `ops.ingestion_completion`
WHERE date = CURRENT_DATE()
  AND status != 'completed'
  AND source IN ('databento_zl', 'databento_other', 'fred', 'scrapecreators_news', 'scrapecreators_trump', 'usda', 'cftc', 'eia', 'weather');
```

---

## ✅ Summary

### Scheduler Architecture:

- ✅ **Separate schedulers** for each data source (parallel)
- ✅ **Coordination** via completion flags in BigQuery
- ✅ **Sequential** for Dataform (features → daily_ml_matrix)
- ✅ **Failure handling** with retries and backoff

### Workflow:

1. **Ingestion** (parallel): All data sources ingest independently
2. **Staging** (parallel): Dataform transforms each source independently
3. **Features** (sequential): Dataform computes features after all ingestion complete
4. **Daily ML Matrix** (sequential): Dataform builds master join after features complete

---

**Last Updated**: November 28, 2025

