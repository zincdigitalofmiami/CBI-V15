# GCP Setup Guide for CBI-V15

**Date**: November 28, 2025  
**Project**: `cbi-v15`  
**Region**: `us-central1` (CRITICAL - no multi-region!)

---

## Prerequisites

- Google Cloud SDK installed (`gcloud`)
- Billing account enabled
- Authenticated: `gcloud auth login`

---

## Step 1: Run Automated Setup Script

```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/setup_gcp_project.sh
```

**What it does**:
- Creates GCP project `cbi-v15` (if needed)
- Enables required APIs (BigQuery, Secret Manager, Cloud Scheduler, etc.)
- Creates 8 BigQuery datasets (raw, staging, features, training, forecasts, api, reference, ops)
- Creates service account for Dataform/Cloud Scheduler
- Grants necessary permissions

**Manual step required**: Link billing account when prompted

---

## Step 2: Store API Keys

### Option A: Automated Script

```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/store_api_keys.sh
```

Choose:
- **Option 3** (Both) - Recommended
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
# Create secrets
echo -n "YOUR_KEY" | gcloud secrets create databento-api-key --data-file=- --project=cbi-v15
echo -n "YOUR_KEY" | gcloud secrets create fred-api-key --data-file=- --project=cbi-v15
echo -n "YOUR_KEY" | gcloud secrets create scrapecreators-api-key --data-file=- --project=cbi-v15
echo -n "YOUR_KEY" | gcloud secrets create glide-api-key --data-file=- --project=cbi-v15
```

---

## Step 3: Verify Setup

```bash
cd /Users/zincdigital/CBI-V15
python scripts/setup/verify_connections.py
```

**Expected output**:
- ✅ GCP Project accessible
- ✅ All 8 BigQuery datasets exist
- ✅ Secret Manager accessible
- ✅ API keys found (if stored)

---

## Step 4: Initialize Dataform

```bash
cd /Users/zincdigital/CBI-V15/dataform
npm install -g @dataform/cli
npm install
dataform init
dataform compile
```

---

## Manual Steps (if needed)

### Create Project Manually

```bash
gcloud projects create cbi-v15 --name="CBI-V15 Soybean Oil Forecasting"
gcloud config set project cbi-v15
```

### Link Billing Account

```bash
gcloud billing projects link cbi-v15 --billing-account=YOUR_BILLING_ACCOUNT_ID
```

Or via console: https://console.cloud.google.com/billing

### Create Datasets Manually

```bash
bq mk --dataset --location=us-central1 --description="Raw source data" cbi-v15:raw
bq mk --dataset --location=us-central1 --description="Cleaned normalized data" cbi-v15:staging
bq mk --dataset --location=us-central1 --description="Engineered features" cbi-v15:features
bq mk --dataset --location=us-central1 --description="Training-ready tables" cbi-v15:training
bq mk --dataset --location=us-central1 --description="Model predictions" cbi-v15:forecasts
bq mk --dataset --location=us-central1 --description="Public API views" cbi-v15:api
bq mk --dataset --location=us-central1 --description="Reference tables" cbi-v15:reference
bq mk --dataset --location=us-central1 --description="Operations monitoring" cbi-v15:ops
```

---

## Cost Considerations

**⚠️ CRITICAL**: All resources in `us-central1` only!

- BigQuery: First 1 TB/month free, then $5/TB
- Secret Manager: $0.06/secret/month
- Cloud Scheduler: First 3 jobs free, then $0.10/job/month
- Cloud Functions: Free tier (2M invocations/month)

**Estimated monthly cost**: <$10 for typical usage

---

## Troubleshooting

### "Permission denied"
- Check: `gcloud auth list`
- Re-authenticate: `gcloud auth login`

### "Billing not enabled"
- Link billing account (see Step 4 above)

### "Dataset already exists"
- This is OK - script will skip existing datasets

### "API not enabled"
- Run: `gcloud services enable bigquery.googleapis.com` (and others)

---

## Next Steps

After GCP setup:
1. ✅ Initialize Dataform
2. ✅ Create first Dataform definitions
3. ✅ Test data ingestion
4. ✅ Export training data
5. ✅ Train first model

---

**Last Updated**: November 28, 2025

