# Complete Setup Execution Guide - CBI-V15

**Date**: November 28, 2025  
**Status**: ✅ **100% READY** - All scripts and documentation complete  
**Folder**: App Development (`568609080192`)  
**Project**: `cbi-v15`  
**Location**: `us-central1` ONLY

---

## 🎯 Complete Setup Sequence

### Prerequisites Checklist

- [ ] ✅ Google Cloud SDK installed (`gcloud`)
- [ ] ✅ BigQuery CLI installed (`bq`)
- [ ] ✅ Python 3.9+ installed
- [ ] ✅ Node.js 18+ installed (for Dataform)
- [ ] ✅ GCP billing account enabled
- [ ] ✅ Authenticated: `gcloud auth login`
- [ ] ✅ Organization admin access (for folder permissions, if needed)

---

## Step 1: Pre-Flight Check

**Run**:
```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/pre_flight_check.sh
```

**What it does**:
- ✅ Verifies Python 3, BigQuery CLI, Google Cloud SDK
- ✅ Sets GCP project to `cbi-v15`
- ✅ Verifies authentication
- ✅ Enables BigQuery API if needed
- ✅ Installs Python dependencies if needed
- ✅ Verifies permissions

**Expected Output**: All checks pass ✅

---

## Step 2: GCP Project Setup

**Run**:
```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/setup_gcp_project.sh
```

**What it does**:
1. ✅ Creates `cbi-v15` project under App Development folder (`568609080192`)
2. ✅ Enables required APIs (BigQuery, Dataform, Secret Manager, Cloud Scheduler, etc.)
3. ✅ Creates 8 BigQuery datasets in `us-central1`:
   - `raw`, `staging`, `features`, `training`, `forecasts`, `signals`, `reference`, `api`, `ops`
4. ✅ Prompts for billing account linking (manual step)

**Expected Time**: ~2-3 minutes

**Manual Step**: Link billing account when prompted

---

## Step 3: IAM Permissions Setup

**Run**:
```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/setup_iam_permissions.sh
```

**What it does**:
1. ✅ Verifies project is in App Development folder
2. ✅ Creates 3 service accounts:
   - `cbi-v15-dataform` (Dataform ETL)
   - `cbi-v15-functions` (Cloud Functions)
   - `cbi-v15-run` (Cloud Run)
3. ✅ Grants project-level permissions
4. ✅ Sets up dataset-level permissions (all 8 datasets)
5. ✅ Configures Cloud Scheduler permissions

**Expected Time**: ~1-2 minutes

**Note**: Folder-level permissions may require org admin approval (script will note this)

---

## Step 4: BigQuery Skeleton Structure

**Run**:
```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/setup_bigquery_skeleton.sh
```

**What it does**:
1. ✅ Creates 8 datasets (if not already created)
2. ✅ Creates 42 skeleton tables (partitioned, clustered)
3. ✅ Initializes reference tables:
   - Regime calendar
   - Train/val/test splits
   - Neural drivers
   - Ingestion completion tracking
4. ✅ Verifies setup (all checks pass)

**Expected Time**: ~2-3 minutes

---

## Step 5: Store API Keys

### Option A: Automated Script

```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/store_api_keys.sh
```

Choose **Option 3** (Both):
- macOS Keychain: For local Python scripts
- Secret Manager: For Cloud Scheduler jobs

### Option B: Manual Setup

#### macOS Keychain (for local scripts):
```bash
security add-generic-password -a databento -s DATABENTO_API_KEY -w YOUR_KEY
security add-generic-password -a fred -s FRED_API_KEY -w YOUR_KEY
security add-generic-password -a scrapecreators -s SCRAPECREATORS_API_KEY -w YOUR_KEY
security add-generic-password -a glide -s GLIDE_API_KEY -w YOUR_KEY
```

#### Secret Manager (for Cloud Scheduler):
```bash
echo -n "YOUR_KEY" | gcloud secrets create databento-api-key --data-file=- --project=cbi-v15
echo -n "YOUR_KEY" | gcloud secrets create fred-api-key --data-file=- --project=cbi-v15
echo -n "YOUR_KEY" | gcloud secrets create scrapecreators-api-key --data-file=- --project=cbi-v15
echo -n "YOUR_KEY" | gcloud secrets create glide-api-key --data-file=- --project=cbi-v15
```

---

## Step 6: Verify Setup

**Run**:
```bash
cd /Users/zincdigital/CBI-V15
python3 scripts/setup/verify_connections.py
```

**Expected Output**:
- ✅ GCP Project accessible
- ✅ All 8 BigQuery datasets exist
- ✅ Secret Manager accessible
- ✅ API keys found (if stored)

---

## Step 7: Initialize Dataform

**Run**:
```bash
cd /Users/zincdigital/CBI-V15/dataform
npm install -g @dataform/cli
npm install
dataform init
dataform compile
```

**Expected Output**: Dataform compiles without errors

---

## Step 8: Test Data Ingestion (Optional)

**Run one ingestion script to verify**:
```bash
cd /Users/zincdigital/CBI-V15
python3 src/ingestion/databento/collect_daily.py
```

**Verify**: Data loads to `raw.databento_futures_ohlcv_1d`

---

## ✅ Success Criteria

**Setup Complete When**:
- ✅ All 8 datasets exist in `us-central1`
- ✅ All 42 tables exist with proper partitioning/clustering
- ✅ Reference tables populated
- ✅ 3 service accounts created
- ✅ Permissions granted (project and dataset level)
- ✅ API keys stored (Keychain and/or Secret Manager)
- ✅ Dataform compiles without errors
- ✅ Verification script passes all checks

---

## 📋 Quick Reference

### All-in-One Execution (if you're confident):

```bash
cd /Users/zincdigital/CBI-V15

# 1. Pre-flight check
./scripts/setup/pre_flight_check.sh

# 2. GCP project setup
./scripts/setup/setup_gcp_project.sh

# 3. IAM permissions
./scripts/setup/setup_iam_permissions.sh

# 4. BigQuery skeleton
./scripts/setup/setup_bigquery_skeleton.sh

# 5. Store API keys
./scripts/setup/store_api_keys.sh

# 6. Verify
python3 scripts/setup/verify_connections.py

# 7. Dataform
cd dataform && npm install && dataform compile
```

**Total Time**: ~10-15 minutes (excluding manual steps)

---

## 🐛 Troubleshooting

### Error: "Project not found"
**Solution**: Run `setup_gcp_project.sh` first

### Error: "Permission denied"
**Solution**: 
1. Run `setup_iam_permissions.sh`
2. Request folder-level permissions from org admin (if needed)

### Error: "Dataset already exists"
**Solution**: This is OK - script handles existing datasets gracefully

### Error: "Billing account not linked"
**Solution**: Link billing account manually:
```bash
gcloud billing projects link cbi-v15 --billing-account=YOUR_BILLING_ACCOUNT_ID
```

---

## 📚 Documentation

- **GCP Setup**: `docs/setup/GCP_SETUP.md`
- **IAM Permissions**: `docs/setup/IAM_PERMISSIONS_GUIDE.md`
- **BigQuery Setup**: `docs/setup/BIGQUERY_SETUP_EXECUTION.md`
- **Quick Start**: `README_BIGQUERY_SETUP.md`

---

## 🎯 Next Steps After Setup

1. ✅ Test data ingestion (one source)
2. ✅ Build first feature table (`staging.market_daily`)
3. ✅ Export training data
4. ✅ Train LightGBM baselines

---

**Status**: ✅ **READY TO EXECUTE**

**Last Updated**: November 28, 2025

