# Next Steps Execution Plan

**Date**: November 28, 2025  
**Status**: Dataform Connected ✅

---

## Current Status

- ✅ GCP Project: `cbi-v15` created
- ✅ BigQuery Datasets: All created
- ✅ Dataform Repository: Connected to GitHub
- ✅ SSH Keys: Configured
- ⏳ API Keys: Need to be stored
- ⏳ Data Ingestion: Not started
- ⏳ Dataform Compilation: Ready to test

---

## Immediate Next Steps

### Step 1: Store API Keys

**Required Keys:**
- Databento API key (critical - for price data)
- ScrapeCreators API key (critical - for Trump/news data)
- FRED API key (optional - for economic data)
- Glide API key (for Vegas Intel)

**Execute:**
```bash
./scripts/setup/store_api_keys.sh
```

**Verify:**
```bash
./scripts/setup/verify_api_keys.sh
```

---

### Step 2: Test Dataform Compilation

**From CLI:**
```bash
cd dataform
npx dataform compile
```

**Expected:**
- 18 actions compiled
- 2 warnings about UDF includes (non-critical)

**Or in UI:**
- Go to: https://console.cloud.google.com/dataform?project=cbi-v15
- Click "Compile" button

---

### Step 3: First Data Ingestion

**Start with Databento (price data):**

```bash
python3 src/ingestion/databento/collect_daily.py
```

**What it does:**
- Collects daily OHLCV for ZL, ZS, ZM, CL, HO, FCPO
- Loads to `raw.databento_futures_ohlcv_1d`
- Handles incremental updates

**Verify:**
```bash
python3 scripts/ingestion/check_data_availability.py
```

---

### Step 4: Run Dataform Staging Layer

**After raw data exists:**

```bash
cd dataform
npx dataform run --tags staging
```

**Or in UI:**
- Click "Run" → Select "staging" tag

**What it builds:**
- `staging.market_daily` - Cleaned market data
- `staging.fred_macro_clean` - Cleaned FRED data
- `staging.news_bucketed` - Aggregated news buckets

---

### Step 5: Run Dataform Features Layer

**After staging data exists:**

```bash
npx dataform run --tags features
```

**What it builds:**
- All feature tables
- `features.daily_ml_matrix` - Master feature table
- Technical indicators
- Cross-asset correlations
- Lagged features

---

### Step 6: Run Data Quality Assertions

**Verify data quality:**

```bash
npx dataform test
```

**Or in UI:**
- Click "Test" button

---

## Priority Order

1. **Store API Keys** (5 min)
2. **Test Compilation** (2 min)
3. **First Ingestion** (Databento - 5 min)
4. **Run Staging** (2 min)
5. **Run Features** (5 min)
6. **Run Assertions** (2 min)

**Total Time**: ~20 minutes for initial setup

---

## Monitoring

**Check System Status:**
```bash
./scripts/system_status.sh
```

**Check Ingestion Status:**
```bash
python3 scripts/ingestion/ingestion_status.py
```

**Check Data Availability:**
```bash
python3 scripts/ingestion/check_data_availability.py
```

---

## Success Criteria

After completing all steps:

- ✅ API keys stored
- ✅ Dataform compiles successfully
- ✅ Raw data flowing into BigQuery
- ✅ Staging tables populated
- ✅ Feature tables populated
- ✅ Assertions passing
- ✅ System operational

---

**Ready to proceed with Step 1: Store API Keys**
