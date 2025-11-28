# CBI-V15 Deployment Guide

**Complete step-by-step deployment instructions**

---

## Prerequisites

- ✅ GCP Project `cbi-v15` created
- ✅ Billing account linked
- ✅ `gcloud` CLI authenticated
- ✅ Python 3.12+ installed
- ✅ Node.js installed (for Dataform)

---

## Phase 1: Initial Setup (Already Complete ✅)

- [x] GCP project created
- [x] BigQuery datasets created
- [x] BigQuery tables created
- [x] Reference data populated
- [x] IAM permissions configured
- [x] APIs enabled
- [x] Dataform structure created
- [x] Code committed to GitHub

---

## Phase 2: Connect Dataform (Manual - UI)

### Steps:

1. **Go to Google Cloud Console**
   - Navigate to: **Dataform** → **Repositories**

2. **Click "Connect Repository"**

3. **Repository Settings:**
   - **Repository**: `zincdigital/CBI-V15`
   - **Branch**: `main`
   - **Root Directory**: `dataform/` ⚠️ **CRITICAL**
   - Click **"Connect"**

4. **Verify Connection:**
   - "Create repository" button should disappear
   - SQL files should be visible
   - Can compile from UI

---

## Phase 3: Store API Keys

### Run Setup Script:
```bash
./scripts/setup/store_api_keys.sh
```

### Keys Needed:
- **Databento API Key** (required)
- **ScrapeCreators API Key** (required)
- **FRED API Key** (optional)
- **Glide API Key** (for Vegas Intel)

### Storage Locations:
- **macOS Keychain**: For local Python scripts
- **GCP Secret Manager**: For Cloud Scheduler/Cloud Functions

---

## Phase 4: Verify Setup

### Run Verification Script:
```bash
./scripts/deployment/verify_deployment.sh
```

**Expected Results:**
- ✅ All datasets exist
- ✅ Service accounts configured
- ✅ APIs enabled
- ⚠️ API keys may not be stored yet (expected)

---

## Phase 5: Test First Ingestion

### Test Databento Ingestion:
```bash
python3 src/ingestion/databento/collect_daily.py
```

**Verify:**
- Data loads to `raw.databento_futures_ohlcv_1d`
- Check BigQuery for new rows
- Verify data quality

---

## Phase 6: Run Dataform Transformations

### Compile Dataform:
```bash
cd dataform
npx dataform compile
```

**Expected:** No errors, 18+ actions compiled

### Run Staging Layer:
```bash
npx dataform run --tags staging
```

**Builds:**
- `staging.market_daily`
- `staging.fred_macro_clean`
- `staging.news_bucketed`

### Run Features Layer:
```bash
npx dataform run --tags features
```

**Builds:**
- All feature tables
- `features.daily_ml_matrix`

### Run Assertions:
```bash
npx dataform test
```

**Verifies:**
- No null keys
- Unique constraints
- Data freshness

---

## Phase 7: Export Training Data

### Run Export Script:
```bash
python3 scripts/export/export_training_data.py
```

**Exports:**
- Training split → Parquet
- Validation split → Parquet
- Test split → Parquet

**Location:** `/Volumes/Satechi Hub/Projects/CBI-V15/data/training_splits/`

---

## Phase 8: Train Baseline Models

### Run Training:
```bash
python3 src/training/baselines/lightgbm_zl.py
```

**Trains:**
- 1w horizon model
- 1m horizon model
- 3m horizon model
- 6m horizon model

**Saves:** Models to `/Volumes/Satechi Hub/Projects/CBI-V15/models/baselines/`

---

## Phase 9: Set Up Automation (Optional)

### Create Cloud Scheduler Jobs:
```bash
./scripts/deployment/create_cloud_scheduler_jobs.sh
```

**Creates:**
- Databento hourly ingestion
- FRED daily ingestion
- ScrapeCreators 4-hourly ingestion
- Dataform staging daily run
- Dataform features daily run

**Note:** Cloud Functions need to be deployed first (if using)

---

## Phase 10: Monitor & Maintain

### Daily Checks:
- Verify ingestion completion
- Check Dataform run status
- Monitor data quality assertions
- Review BigQuery costs

### Weekly Checks:
- Review model performance
- Check for data gaps
- Verify feature completeness
- Update documentation

---

## Troubleshooting

### Dataform Connection Issues
- Verify Root Directory is `dataform/`
- Check GitHub repository access
- Verify service account permissions

### Ingestion Failures
- Check API keys in Keychain/Secret Manager
- Verify BigQuery table permissions
- Check API rate limits

### Dataform Compilation Errors
- Review SQL syntax
- Check table dependencies
- Verify variable references

---

## Success Criteria

- [x] Infrastructure complete
- [ ] Dataform connected to GitHub
- [ ] API keys stored
- [ ] First ingestion successful
- [ ] Dataform transformations running
- [ ] Training data exported
- [ ] Baseline models trained
- [ ] Automation configured (optional)

---

## Quick Reference

**Test Connections:**
```bash
python3 scripts/ingestion/test_connections.py
```

**Verify Setup:**
```bash
./scripts/deployment/verify_deployment.sh
```

**Store API Keys:**
```bash
./scripts/setup/store_api_keys.sh
```

**Compile Dataform:**
```bash
cd dataform && npx dataform compile
```

---

**Last Updated**: November 28, 2025

