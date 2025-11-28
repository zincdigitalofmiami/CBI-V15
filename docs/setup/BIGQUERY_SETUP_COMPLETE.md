# BigQuery Setup - Complete Guide

**Date**: November 28, 2025  
**Status**: ✅ **READY FOR EXECUTION**

---

## 🎯 Setup Steps

### Step 1: Create BigQuery Datasets

**Script**: `scripts/setup/create_bigquery_datasets.py`

**What it does**:
- Creates 8 datasets in `us-central1`:
  - `raw` - Raw source data
  - `staging` - Cleaned data
  - `features` - Engineered features
  - `training` - Training-ready tables
  - `forecasts` - Model predictions
  - `api` - Public API views
  - `reference` - Reference tables
  - `ops` - Operations monitoring

**Run**:
```bash
python3 scripts/setup/create_bigquery_datasets.py
```

---

### Step 2: Create Skeleton Tables

**Script**: `scripts/setup/create_skeleton_tables.sql`

**What it does**:
- Creates all skeleton tables with proper partitioning/clustering
- No joins, just table structure
- All tables partitioned by `DATE(date)`
- Clustered by `symbol` (where applicable)

**Run**:
```bash
bq query --use_legacy_sql=false --project_id=cbi-v15 < scripts/setup/create_skeleton_tables.sql
```

---

### Step 3: Run Complete Setup

**Script**: `scripts/setup/setup_bigquery_skeleton.sh`

**What it does**:
- Runs Step 1 (create datasets)
- Runs Step 2 (create skeleton tables)
- Verifies structure

**Run**:
```bash
./scripts/setup/setup_bigquery_skeleton.sh
```

---

## ✅ Verification

After setup, verify structure:

```bash
# List all datasets
bq ls --project_id=cbi-v15

# List tables in a dataset
bq ls --project_id=cbi-v15 raw
bq ls --project_id=cbi-v15 staging
bq ls --project_id=cbi-v15 features
bq ls --project_id=cbi-v15 training
bq ls --project_id=cbi-v15 forecasts
```

---

## 📊 Table Structure Summary

### Raw Layer (7 tables)
- `databento_futures_ohlcv_1d`
- `fred_economic`
- `usda_reports`
- `cftc_cot`
- `eia_biofuels`
- `weather_noaa`
- `scrapecreators_trump`

### Staging Layer (7 tables)
- `market_daily`
- `fred_macro_clean`
- `usda_reports_clean`
- `cftc_positions`
- `eia_biofuels_clean`
- `weather_regions_aggregated`
- `trump_policy_intelligence`

### Features Layer (7 tables)
- `technical_indicators_us_oil_solutions` (19 features)
- `fx_indicators_daily` (16 features)
- `fundamental_spreads_daily` (5 features)
- `pair_correlations_daily` (112 features)
- `cross_asset_betas_daily` (28 features)
- `lagged_features_daily` (96 features)
- `daily_ml_matrix` (276 features - master join)

### Training Layer (4 tables)
- `zl_training_1w`
- `zl_training_1m`
- `zl_training_3m`
- `zl_training_6m`

### Forecasts Layer (4 tables)
- `zl_predictions_1w`
- `zl_predictions_1m`
- `zl_predictions_3m`
- `zl_predictions_6m`

**Total**: 29 skeleton tables ✅

---

## 🎯 Next Steps

After skeleton structure is created:

1. ✅ **USDA Ingestion** (REQUIRED before baselines)
   - Implement `src/ingestion/usda/collect_usda_comprehensive.py`
   - Load to `raw.usda_reports`

2. ✅ **CFTC Ingestion** (REQUIRED before baselines)
   - Implement `src/ingestion/cftc/collect_cftc_comprehensive.py`
   - Load to `raw.cftc_cot`

3. ✅ **EIA Ingestion** (REQUIRED before baselines)
   - Implement `src/ingestion/eia/collect_eia_comprehensive.py`
   - Load to `raw.eia_biofuels`

4. ✅ **Build Dataform Feature Tables**
   - Implement feature calculations in Dataform
   - Build `daily_ml_matrix` (master join)

5. ✅ **Export Training Data**
   - Export from `daily_ml_matrix` to Parquet
   - Begin baseline training

---

**Last Updated**: November 28, 2025

