# BigQuery Setup - Ready to Execute

**Date**: November 28, 2025  
**Status**: ✅ **100% READY** - All prerequisites met, scripts ready

---

## ✅ Prerequisites Complete

- [x] ✅ All 42 tables accounted for (skeleton structure complete)
- [x] ✅ All math validated (institutional-grade)
- [x] ✅ Sentiment logic corrected (China/Tariffs fixed)
- [x] ✅ Pre-built tools evaluated (5 approved, no bloat)
- [x] ✅ Validation schema created (Pandera)
- [x] ✅ Setup scripts ready

---

## 🚀 Execute BigQuery Setup

### Single Command (Complete Setup):

```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/setup_bigquery_skeleton.sh
```

**What it does**:
1. Creates 8 datasets in `us-central1`
2. Creates 42 skeleton tables (partitioned, clustered)
3. Initializes reference tables (regime calendar, splits, neural drivers)
4. Verifies setup (all checks pass)

**Expected Time**: ~2-3 minutes

---

### Manual Steps (If Needed):

#### Step 1: Create Datasets
```bash
python3 scripts/setup/create_bigquery_datasets.py
```

#### Step 2: Create Skeleton Tables
```bash
bq query --project_id=cbi-v15 --location=us-central1 --use_legacy_sql=false < scripts/setup/create_skeleton_tables.sql
```

#### Step 3: Initialize Reference Tables
```bash
bq query --project_id=cbi-v15 --location=us-central1 --use_legacy_sql=false < scripts/setup/initialize_reference_tables.sql
```

#### Step 4: Verify Setup
```bash
python3 scripts/setup/verify_bigquery_setup.py
```

---

## ✅ Success Criteria

**Setup Complete When**:
- ✅ All 8 datasets exist in `us-central1`
- ✅ All 42 tables exist with proper partitioning/clustering
- ✅ Reference tables populated (regime calendar, splits, neural drivers)
- ✅ Ops table ready (ingestion completion tracking)
- ✅ Verification script passes all checks

---

## 📋 After Setup

1. ✅ **Test Data Ingestion**
   - Run `src/ingestion/databento/collect_daily.py`
   - Verify data loads to `raw.databento_futures_ohlcv_1d`

2. ✅ **Test Dataform Compilation**
   - Run `cd dataform && dataform compile`
   - Verify no errors

3. ✅ **Build First Feature Table**
   - Run Dataform transformation for `staging.market_daily`
   - Verify data quality

---

**Ready to Execute**: ✅ **YES** - Run `./scripts/setup/setup_bigquery_skeleton.sh`

---

**Last Updated**: November 28, 2025

