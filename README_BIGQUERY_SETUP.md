# BigQuery Setup - Quick Start Guide

**Date**: November 28, 2025  
**Project**: cbi-v15  
**Location**: us-central1 ONLY

---

## 🚀 Quick Start

### Step 1: Pre-Flight Check

```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/pre_flight_check.sh
```

**What it checks**:
- ✅ Python 3 installed
- ✅ BigQuery CLI (bq) installed
- ✅ Google Cloud SDK (gcloud) installed
- ✅ GCP project set to `cbi-v15`
- ✅ GCP authentication active
- ✅ BigQuery API enabled
- ✅ Python dependencies installed
- ✅ Setup scripts exist
- ✅ BigQuery permissions verified

**If any check fails**: The script will guide you to fix it.

---

### Step 2: Execute BigQuery Setup

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

**Expected Output**:
```
🚀 Starting BigQuery skeleton setup for project: cbi-v15 in location: us-central1

📊 Step 1: Creating BigQuery datasets...
✅ Dataset 'raw' created in us-central1
✅ Dataset 'staging' created in us-central1
... (8 datasets total)

📋 Step 2: Creating skeleton tables...
✅ Skeleton tables created successfully

📚 Step 3: Initializing reference tables...
✅ Reference tables initialized successfully

🔍 Step 4: Verifying BigQuery setup...
✅ All 8 datasets verified
✅ All 42 tables verified
✅ Reference data verified

✅ BigQuery skeleton setup complete!
```

---

## 📋 Prerequisites

### Required Tools:

1. **Python 3** (3.9+)
   ```bash
   python3 --version
   ```

2. **Google Cloud SDK**
   ```bash
   gcloud --version
   ```

3. **BigQuery CLI**
   ```bash
   bq version
   ```

### Required Setup:

1. **GCP Project Created**
   - Project ID: `cbi-v15`
   - Billing enabled

2. **APIs Enabled**
   - BigQuery API
   - (Others enabled automatically by setup script)

3. **Authentication**
   ```bash
   gcloud auth login
   gcloud config set project cbi-v15
   ```

4. **Python Dependencies**
   ```bash
   pip3 install google-cloud-bigquery
   ```

---

## ✅ Verification

After setup, verify everything worked:

```bash
# List datasets
bq ls --project_id=cbi-v15

# List tables in a dataset
bq ls --project_id=cbi-v15 raw

# Check table structure
bq show --project_id=cbi-v15 raw.databento_futures_ohlcv_1d

# Run verification script
python3 scripts/setup/verify_bigquery_setup.py
```

---

## 🐛 Troubleshooting

### Error: "Project not found"
```bash
gcloud config set project cbi-v15
gcloud projects list  # Verify project exists
```

### Error: "Permission denied"
```bash
# Check your IAM roles
gcloud projects get-iam-policy cbi-v15

# Ensure you have BigQuery Admin or Editor role
```

### Error: "BigQuery API not enabled"
```bash
gcloud services enable bigquery.googleapis.com --project=cbi-v15
```

### Error: "Dataset already exists"
- This is OK - the script handles existing datasets gracefully
- Tables will be created if they don't exist

---

## 📋 Next Steps After Setup

1. ✅ **Test Data Ingestion**
   - Run one ingestion script (e.g., Databento)
   - Verify data loads to `raw.databento_futures_ohlcv_1d`

2. ✅ **Test Dataform Compilation**
   ```bash
   cd dataform
   dataform compile
   ```

3. ✅ **Build First Feature Table**
   - Run Dataform transformation for `staging.market_daily`
   - Verify data quality

4. ✅ **Validate with Pandera**
   - Run validation schema on sample data
   - Verify no logic inversions

---

## 📚 Documentation

- **Execution Guide**: `docs/setup/BIGQUERY_SETUP_EXECUTION.md`
- **GCP Setup**: `docs/setup/GCP_SETUP.md`
- **Status**: `EXECUTION_STATUS.md`

---

**Last Updated**: November 28, 2025

