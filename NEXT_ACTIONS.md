# CBI-V15 Next Actions

**Date**: November 28, 2025  
**Status**: ✅ Infrastructure Complete - Ready for Data Ingestion

---

## ✅ Completed

1. ✅ GCP Project (`cbi-v15`) created
2. ✅ BigQuery datasets (8) created
3. ✅ BigQuery tables (42) created and populated
4. ✅ Reference data initialized
5. ✅ IAM permissions configured
6. ✅ APIs enabled
7. ✅ Dataform structure created and compiles
8. ✅ GitHub repository ready

---

## 🎯 Immediate Next Steps

### 1. Connect Dataform to GitHub (Manual - UI)

**Action Required:**
- Go to Google Cloud Console → Dataform
- Click "Connect Repository"
- Repository: `zincdigital/CBI-V15`
- Branch: `main`
- **Root Directory: `dataform/`** ⚠️ Critical
- Click "Connect"

**Expected Result:**
- "Create repository" button disappears
- SQL files visible in Dataform UI
- Can compile/run from UI

---

### 2. Store API Keys

**Script Available:** `scripts/setup/store_api_keys.sh`

**Keys Needed:**
- Databento API key
- ScrapeCreators API key
- FRED API key (if using)
- Other data source keys

**Storage:**
- **GCP Secret Manager**: For Cloud Scheduler/Cloud Functions
- **macOS Keychain**: For local Python scripts

**Run:**
```bash
./scripts/setup/store_api_keys.sh
```

---

### 3. Test First Data Ingestion

**Recommended First Test:** Databento (price data)

**Steps:**
1. Verify API key is stored
2. Run ingestion script:
   ```bash
   python src/ingestion/databento/collect_daily.py
   ```
3. Verify data loads to `raw.databento_futures_ohlcv_1d`
4. Check BigQuery for new rows

---

### 4. Run Dataform Transformations

**After data is ingested:**

1. **Compile Dataform** (verify no errors):
   ```bash
   cd dataform
   npx dataform compile
   ```

2. **Run Staging Layer**:
   ```bash
   npx dataform run --tags staging
   ```
   - Builds `staging.market_daily`
   - Builds `staging.fred_macro_clean`
   - Builds `staging.news_bucketed`

3. **Run Feature Layer**:
   ```bash
   npx dataform run --tags features
   ```
   - Builds all feature tables
   - Creates `features.daily_ml_matrix`

4. **Run Assertions**:
   ```bash
   npx dataform test
   ```
   - Verifies data quality
   - Checks for nulls, duplicates, freshness

---

### 5. Export Training Data

**After features are built:**

```bash
python scripts/export/export_training_data.py
```

**Exports:**
- `training.daily_ml_matrix_train` → Parquet
- `training.daily_ml_matrix_val` → Parquet
- `training.daily_ml_matrix_test` → Parquet

**Location:** `/Volumes/Satechi Hub/Projects/CBI-V15/data/training_splits/`

---

### 6. Train Baseline Models

**After data is exported:**

```bash
python src/training/baselines/lightgbm_zl.py
```

**Trains:**
- 1w horizon model
- 1m horizon model
- 3m horizon model
- 6m horizon model

**Saves:** Models to `/Volumes/Satechi Hub/Projects/CBI-V15/models/baselines/`

---

## 📋 Checklist

### Setup Phase
- [ ] Connect Dataform to GitHub repository
- [ ] Store API keys (Secret Manager + Keychain)
- [ ] Verify API connections

### Data Ingestion Phase
- [ ] Test Databento ingestion
- [ ] Test FRED ingestion
- [ ] Test ScrapeCreators ingestion
- [ ] Verify data quality

### ETL Phase
- [ ] Compile Dataform (verify no errors)
- [ ] Run staging layer transformations
- [ ] Run feature layer transformations
- [ ] Run assertions (verify data quality)

### Training Phase
- [ ] Export training data splits
- [ ] Train LightGBM baseline models
- [ ] Evaluate model performance
- [ ] Save models

---

## 🔧 Configuration Files

**Key Files to Review:**
- `dataform/dataform.json` - Dataform configuration
- `config/schedulers/ingestion_schedules.yaml` - Ingestion schedules
- `scripts/setup/store_api_keys.sh` - API key storage script

---

## 📊 Current Status

**Infrastructure**: ✅ 100% Complete  
**Dataform**: ✅ Ready (needs GitHub connection)  
**API Keys**: ⚠️ Need to be stored  
**Data Ingestion**: ⚠️ Not started  
**ETL**: ⚠️ Waiting for data  
**Training**: ⚠️ Waiting for features  

---

**Next Action**: Connect Dataform to GitHub, then proceed with API key storage and first ingestion test.

