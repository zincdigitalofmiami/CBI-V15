# BigQuery Setup Execution - Step-by-Step Guide

**Date**: November 28, 2025  
**Status**: ✅ **READY TO EXECUTE** - All prerequisites met  
**Project**: cbi-v15  
**Location**: us-central1 ONLY

---

## ✅ Prerequisites Verified

### Before Starting:

- [x] ✅ All 42 tables accounted for (skeleton structure complete)
- [x] ✅ All math validated (institutional-grade)
- [x] ✅ Sentiment logic corrected (China/Tariffs fixed)
- [x] ✅ Pre-built tools evaluated (5 approved, no bloat)
- [x] ✅ GCP project created (`cbi-v15`)
- [x] ✅ APIs enabled (BigQuery, Dataform, Secret Manager, etc.)
- [x] ✅ Service account created (with proper permissions)

---

## 🎯 Execution Steps

### Step 1: Create BigQuery Datasets

**Script**: `scripts/setup/create_bigquery_datasets.py`

**What it does**:
- Creates 8 datasets in `us-central1`:
  - `raw` - Source data declarations
  - `staging` - Staged, normalized data
  - `features` - Engineered features
  - `training` - Training-ready tables
  - `forecasts` - Model predictions
  - `api` - Public API views
  - `reference` - Reference tables & mappings
  - `ops` - Operations monitoring

**Run**:
```bash
cd /Volumes/Satechi Hub/CBI-V15
python3 scripts/setup/create_bigquery_datasets.py
```

**Expected Output**:
```
✅ Dataset raw created in us-central1
✅ Dataset staging created in us-central1
✅ Dataset features created in us-central1
✅ Dataset training created in us-central1
✅ Dataset forecasts created in us-central1
✅ Dataset api created in us-central1
✅ Dataset reference created in us-central1
✅ Dataset ops created in us-central1
```

---

### Step 2: Create Skeleton Tables

**Script**: `scripts/setup/create_skeleton_tables.sql`

**What it does**:
- Creates 42 skeleton tables with proper partitioning/clustering
- No joins (pure table definitions)
- Partitioned by `DATE(date)`
- Clustered by `symbol` (where applicable)

**Run**:
```bash
cd /Volumes/Satechi Hub/CBI-V15
bq query --project_id=cbi-v15 --location=us-central1 --use_legacy_sql=false < scripts/setup/create_skeleton_tables.sql
```

**Expected Output**:
```
✅ Table raw.databento_futures_ohlcv_1d created
✅ Table raw.fred_economic created
✅ Table raw.scrapecreators_news_buckets created
... (42 tables total)
```

---

### Step 3: Verify Structure

**Verification Queries**:

```sql
-- Check all datasets exist
SELECT schema_name 
FROM `cbi-v15.INFORMATION_SCHEMA.SCHEMATA`
WHERE schema_name IN ('raw', 'staging', 'features', 'training', 'forecasts', 'api', 'reference', 'ops')
ORDER BY schema_name;

-- Check all tables exist (should return 42 rows)
SELECT 
  table_schema,
  table_name,
  partition_column,
  clustering_columns
FROM `cbi-v15.INFORMATION_SCHEMA.TABLES`
WHERE table_schema IN ('raw', 'staging', 'features', 'training', 'forecasts', 'api', 'reference', 'ops')
ORDER BY table_schema, table_name;

-- Verify partitioning
SELECT 
  table_schema,
  table_name,
  partition_column
FROM `cbi-v15.INFORMATION_SCHEMA.TABLES`
WHERE partition_column IS NOT NULL
ORDER BY table_schema, table_name;
```

**Expected**: 42 tables, all partitioned by date, clustered where applicable

---

### Step 4: Initialize Reference Tables

**Create Reference Data**:

```sql
-- Regime Calendar
CREATE OR REPLACE TABLE `cbi-v15.reference.regime_calendar` (
  regime_type STRING,
  start_date DATE,
  end_date DATE,
  description STRING
)
CLUSTER BY regime_type;

INSERT INTO `cbi-v15.reference.regime_calendar` VALUES
('trump_2018', DATE('2018-01-01'), DATE('2020-12-31'), 'Trump first term trade war era'),
('trump_2024', DATE('2024-01-01'), DATE('2025-12-31'), 'Trump second term'),
('normal', DATE('2010-01-01'), DATE('2017-12-31'), 'Normal market conditions'),
('crisis', DATE('2020-03-01'), DATE('2020-06-30'), 'COVID crisis');

-- Train/Val/Test Splits
CREATE OR REPLACE TABLE `cbi-v15.reference.train_val_test_splits` (
  set_name STRING,
  start_date DATE,
  end_date DATE
);

INSERT INTO `cbi-v15.reference.train_val_test_splits` VALUES
('train', DATE('2010-01-01'), DATE('2018-12-31')),
('val', DATE('2019-01-01'), DATE('2021-12-31')),
('test', DATE('2022-01-01'), DATE('2025-12-31'));
```

---

### Step 5: Initialize Ops Table

**Create Ingestion Completion Tracking**:

```sql
CREATE OR REPLACE TABLE `cbi-v15.ops.ingestion_completion` (
  date DATE,
  source STRING,
  completed_at TIMESTAMP,
  status STRING,
  rows_ingested INT64
)
PARTITION BY DATE(date)
CLUSTER BY source;
```

---

## ✅ Verification Checklist

### After Setup:

- [ ] ✅ All 8 datasets created in `us-central1`
- [ ] ✅ All 42 tables created with proper partitioning
- [ ] ✅ Clustering verified (symbol where applicable)
- [ ] ✅ Reference tables populated (regime calendar, splits)
- [ ] ✅ Ops table created (ingestion completion tracking)
- [ ] ✅ No errors in BigQuery console

---

## 📋 Next Steps After BigQuery Setup

1. ✅ **Test Data Ingestion** (one data source)
   - Run `src/ingestion/databento/collect_daily.py`
   - Verify data loads to `raw.databento_futures_ohlcv_1d`

2. ✅ **Test Dataform Compilation**
   - Run `dataform compile` in `dataform/` directory
   - Verify no errors

3. ✅ **Build First Feature Table**
   - Run Dataform transformation for `staging.market_daily`
   - Verify data quality

4. ✅ **Validate with Pandera**
   - Run validation schema on sample data
   - Verify no logic inversions

---

## 🎯 Success Criteria

**BigQuery Setup Complete When**:
- ✅ All 8 datasets exist
- ✅ All 42 tables exist with proper structure
- ✅ Reference tables populated
- ✅ Ops table ready for scheduler coordination
- ✅ No errors in BigQuery console
- ✅ Ready for data ingestion

---

**Last Updated**: November 28, 2025
