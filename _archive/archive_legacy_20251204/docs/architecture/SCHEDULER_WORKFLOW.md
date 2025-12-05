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
- ⚠️ Need coordination for downstream feature / training builds

**Verdict**: ✅ **SEPARATE SCHEDULERS** - Benefits outweigh complexity

---

## 📊 Scheduler Workflow

### Workflow Overview:

```
Cloud Scheduler (per data source)
    ↓
Cloud Function / Cloud Run (ingestion script)
    ↓
BigQuery raw_staging.<source>_<bucket>_<run_id>
    ↓
BigQuery MERGE into raw.<source>_<entity>  -- idempotent on primary key
    ↓
Python feature builders on Mac (or future batch job)
    ↓
BigQuery FEATURES / TRAINING tables (for example training.daily_ml_matrix)
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
5. Upload to BigQuery `raw_staging.databento_daily_<run_id>` and MERGE into `raw.databento_futures_ohlcv_1d` (idempotent on `(symbol, date)`)
6. Optional: refresh `staging.market_daily` via BigQuery SQL view or scheduled query (no Dataform)

**Dependencies**: None (independent)

**Failure Handling**: Retry 3x, max backoff 3600s

---

### 2. FRED Macro Data (Bucketed)

**Schedulers (separate per bucket)**:
- `fred-fx-daily` (daily at 2:00 AM ET)
- `fred-rates-curve-daily` (daily at 2:05 AM ET)
- `fred-financial-conditions-daily` (daily at 2:10 AM ET)

**Workflow (per bucket)**:
1. Cloud Scheduler triggers the corresponding Cloud Function / Cloud Run service
2. The function runs the matching FRED collector:
   - `fred-fx-daily` → `src/ingestion/fred/collect_fred_fx.py`
   - `fred-rates-curve-daily` → `src/ingestion/fred/collect_fred_rates_curve.py` (planned)
   - `fred-financial-conditions-daily` → `src/ingestion/fred/collect_fred_financial_conditions.py` (planned)
3. Each script:
   - Pulls only its own bucket of FRED series (FX, rates/curve, or financial conditions)
   - Writes to a per-run staging table in `raw_staging.*`
   - MERGEs into `raw.fred_economic` on `(series_id, date)` (idempotent, no duplicates)
4. Optional: A separate macro staging job builds `staging.fred_macro_clean` from `raw.fred_economic`

**Dependencies**: None (buckets are independent)

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
9. Upload to BigQuery `raw_staging.scrapecreators_news_buckets_<run_id>` and MERGE into `raw.scrapecreators_news_buckets` on `(bucket, date)`
10. Optional: build `staging.news_bucketed` and `staging.sentiment_buckets` via BigQuery SQL / Python ETL (no Dataform)

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
6. Upload to BigQuery `raw_staging.scrapecreators_trump_<run_id>` and MERGE into `raw.scrapecreators_trump` on `(post_id, created_at DATE)`
7. Optional: build `staging.trump_policy_intelligence` via BigQuery SQL / Python ETL

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
4. Parse and normalize data
5. Upload to BigQuery `raw_staging.<source>_<bucket>_<run_id>`
6. MERGE into canonical `raw.*` table on its primary key (for example `(series_id, date)` or `(symbol, date)`)
7. Optional: build corresponding `staging.*` tables via BigQuery SQL / Python ETL

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
5. Upload to BigQuery `raw_staging.weather_noaa_<run_id>` and MERGE into `raw.weather_noaa` on `(station_id, date)`
6. Optional: build `staging.weather_regions_aggregated` via BigQuery SQL / Python ETL

**Dependencies**: None (independent)

**Failure Handling**: Retry 3x, max backoff 1800s

---

### 7. Feature Computation (Python + BigQuery)

**Scheduler**: `features-daily-build` (PLANNED, daily at 3 AM ET)  
**Current state**: Run manually on Mac using Python scripts.

**Workflow (target state)**:
1. Scheduler (or Mac cron) checks `ops.ingestion_completion` to confirm all ingestion jobs are complete for the day.
2. Run Python feature builders, for example:
   - `src/features/build_daily_ml_matrix.py`
   - `src/features/build_fx_indicators_daily.py`
3. Use loaders (for example `scripts/load_daily_ml_matrix.py`) to write feature/training tables into BigQuery:
   - `features.fx_indicators_daily`
   - `training.daily_ml_matrix`
4. Run QA checks (row counts, nulls, regime coverage) and log results to `ops.*`.

**Dependencies**: All ingestion schedulers (must wait for completion)

**Failure Handling**: Retry 2x, max backoff 3600s (for Cloud jobs); Mac runs are manual and must be monitored locally.

---

### 8. Daily ML Matrix Build (Python-First)

**Scheduler**: `ml-matrix-daily` (PLANNED)  
**Current state**: Triggered manually on Mac.

**Workflow (Python-first)**:
1. Run `python src/features/build_daily_ml_matrix.py` on the Mac to generate `TrainingData/exports/daily_ml_matrix.parquet`.
2. Run `python scripts/load_daily_ml_matrix.py --parquet TrainingData/exports/daily_ml_matrix.parquet --table cbi-v15.training.daily_ml_matrix --write-disposition WRITE_TRUNCATE` to load into BigQuery (MONTH partitioned, clustered by `symbol, regime`).
3. Define or refresh lightweight training views (for example `training.zl_train`, `training.zl_val`, `training.zl_test`) as pure filters on `training.daily_ml_matrix`.
4. Optional future: wrap steps 1–3 in a single Cloud Run / Cloud Scheduler job once Mac-only backfill period is over.

**Dependencies**: Successful feature computation and ingestion completion for the day.

**Failure Handling**: Manual rerun on Mac; future scheduler can use standard retry policies.

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

**Feature-build scheduler checks**:
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
- ✅ **Sequential** for feature builds (features → daily_ml_matrix)
- ✅ **Failure handling** with retries and backoff

### Workflow:

1. **Ingestion** (parallel): All data sources ingest independently into `raw_staging.*` and MERGE into `raw.*`.
2. **Staging** (parallel/optional): Lightweight BigQuery SQL / Python ETL builds `staging.*` panels where needed.
3. **Features** (sequential): Python feature builders compute feature tables (for example `features.fx_indicators_daily`) after ingestion completes.
4. **Daily ML Matrix** (sequential): Python + loaders build the master join `training.daily_ml_matrix` after features are ready.

---

**Last Updated**: November 28, 2025
