# Post-Dataform Connection Guide

**After connecting Dataform to GitHub, follow these steps**

---

## ✅ Connection Complete Checklist

After connecting Dataform in the UI, verify:

- [ ] Repository shows: `CBI-V15`
- [ ] Files visible in `definitions/` folders
- [ ] Can see SQL files (`.sqlx`)
- [ ] "Compile" button works
- [ ] Compilation shows: "Compiled 18 action(s)"

---

## Step 1: Verify Compilation

### In Dataform UI:
1. Click **"Compile"** button
2. Should show: **"Compiled 18 action(s)"**
3. May show 2 warnings about UDF includes (non-critical - can add later)

### Or from CLI:
```bash
cd dataform
npx dataform compile
```

**Expected**: 18 actions compiled successfully

---

## Step 2: Store API Keys

**Before data ingestion, store API keys:**

```bash
./scripts/setup/store_api_keys.sh
```

**Keys to store:**
- Databento API key (required)
- ScrapeCreators API key (required)
- FRED API key (optional)
- Glide API key (for Vegas Intel)

**Verify keys stored:**
```bash
./scripts/setup/verify_api_keys.sh
```

---

## Step 3: First Data Ingestion

**Start with Databento (price data):**

```bash
python3 src/ingestion/databento/collect_daily.py
```

**What it does:**
- Collects daily OHLCV data for ZL, ZS, ZM, CL, HO, FCPO
- Loads to `raw.databento_futures_ohlcv_1d`
- Handles incremental updates

**Verify ingestion:**
```bash
python3 scripts/ingestion/check_data_availability.py
```

Should show data in raw tables.

---

## Step 4: Run Dataform Staging

**After raw data exists:**

### In Dataform UI:
1. Click **"Run"** button
2. Select **"staging"** tag
3. Click **"Run"**

### Or from CLI:
```bash
cd dataform
npx dataform run --tags staging
```

**What it builds:**
- `staging.market_daily` - Cleaned market data
- `staging.fred_macro_clean` - Cleaned FRED data
- `staging.news_bucketed` - Aggregated news buckets

**Verify:**
```bash
python3 scripts/ingestion/check_data_availability.py
```

Should show data in staging tables.

---

## Step 5: Run Dataform Features

**After staging data exists:**

### In Dataform UI:
1. Click **"Run"** button
2. Select **"features"** tag
3. Click **"Run"**

### Or from CLI:
```bash
npx dataform run --tags features
```

**What it builds:**
- All feature tables
- `features.daily_ml_matrix` - Master feature table (276 features)
- Technical indicators
- Cross-asset correlations
- Lagged features
- And more...

**Verify:**
```bash
python3 scripts/ingestion/check_data_availability.py
```

Should show data in feature tables.

---

## Step 6: Run Data Quality Assertions

**Verify data quality:**

### In Dataform UI:
1. Click **"Test"** button
2. Review assertion results

### Or from CLI:
```bash
npx dataform test
```

**Assertions check:**
- No null keys
- Unique constraints
- Data freshness
- Join integrity

---

## Step 7: Export Training Data

**After features are built:**

```bash
python3 scripts/export/export_training_data.py
```

**Exports:**
- Training split → Parquet
- Validation split → Parquet
- Test split → Parquet

**Location**: `/Volumes/Satechi Hub/Projects/CBI-V15/data/training_splits/`

---

## Step 8: Train Baseline Models

**After data is exported:**

```bash
python3 src/training/baselines/lightgbm_zl.py
```

**Trains:**
- 1w horizon model
- 1m horizon model
- 3m horizon model
- 6m horizon model

**Saves**: Models to `/Volumes/Satechi Hub/Projects/CBI-V15/models/baselines/`

---

## 📊 Monitoring

### Check System Status:
```bash
./scripts/system_status.sh
```

### Check Ingestion Status:
```bash
python3 scripts/ingestion/ingestion_status.py
```

### Check Data Availability:
```bash
python3 scripts/ingestion/check_data_availability.py
```

---

## 🎯 Success Criteria

After completing all steps:

- ✅ Data flowing into BigQuery
- ✅ Staging tables populated
- ✅ Feature tables populated
- ✅ Training data exported
- ✅ Baseline models trained
- ✅ System operational

---

**Status**: Ready for post-connection operations!

