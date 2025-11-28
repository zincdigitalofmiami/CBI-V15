#!/bin/bash
# Verify complete deployment status

set -e

PROJECT_ID="cbi-v15"
REGION="us-central1"

echo "🔍 CBI-V15 Deployment Verification"
echo "=================================="
echo ""

# Check GCP Project
echo "1. Checking GCP Project..."
if gcloud projects describe "${PROJECT_ID}" &> /dev/null; then
    echo "   ✅ Project ${PROJECT_ID} exists"
else
    echo "   ❌ Project ${PROJECT_ID} not found"
    exit 1
fi

# Check BigQuery Datasets
echo "2. Checking BigQuery Datasets..."
DATASETS=("raw" "staging" "features" "training" "forecasts" "api" "reference" "ops")
for dataset in "${DATASETS[@]}"; do
    if bq show "${PROJECT_ID}:${dataset}" &> /dev/null; then
        echo "   ✅ Dataset ${dataset} exists"
    else
        echo "   ❌ Dataset ${dataset} missing"
    fi
done

# Check BigQuery Tables (sample)
echo "3. Checking BigQuery Tables..."
TABLE_COUNT=$(bq ls -d "${PROJECT_ID}:raw" --format=json | jq length 2>/dev/null || echo "0")
echo "   📊 Raw dataset: ${TABLE_COUNT} tables"

# Check Service Accounts
echo "4. Checking Service Accounts..."
SERVICE_ACCOUNTS=("cbi-v15-dataform" "cbi-v15-functions" "cbi-v15-run")
for sa in "${SERVICE_ACCOUNTS[@]}"; do
    if gcloud iam service-accounts describe "${sa}@${PROJECT_ID}.iam.gserviceaccount.com" &> /dev/null; then
        echo "   ✅ Service account ${sa} exists"
    else
        echo "   ❌ Service account ${sa} missing"
    fi
done

# Check APIs
echo "5. Checking Enabled APIs..."
APIS=("bigquery.googleapis.com" "dataform.googleapis.com" "cloudscheduler.googleapis.com" "secretmanager.googleapis.com")
for api in "${APIS[@]}"; do
    if gcloud services list --enabled --project="${PROJECT_ID}" --filter="name:${api}" --format="value(name)" | grep -q "${api}"; then
        echo "   ✅ API ${api} enabled"
    else
        echo "   ⚠️  API ${api} not enabled"
    fi
done

# Check Dataform Repository
echo "6. Checking Dataform Repository..."
if gcloud dataform repositories describe CBI-V15 --location="${REGION}" --project="${PROJECT_ID}" &> /dev/null; then
    echo "   ✅ Dataform repository connected"
else
    echo "   ⚠️  Dataform repository not connected (connect in UI)"
fi

# Check Cloud Scheduler Jobs
echo "7. Checking Cloud Scheduler Jobs..."
JOB_COUNT=$(gcloud scheduler jobs list --project="${PROJECT_ID}" --location="${REGION}" --format="value(name)" 2>/dev/null | wc -l || echo "0")
echo "   📊 ${JOB_COUNT} scheduler jobs configured"

# Check API Keys (Secret Manager)
echo "8. Checking API Keys in Secret Manager..."
SECRETS=("databento-api-key" "scrapecreators-api-key" "fred-api-key")
for secret in "${SECRETS[@]}"; do
    if gcloud secrets describe "${secret}" --project="${PROJECT_ID}" &> /dev/null; then
        echo "   ✅ Secret ${secret} exists"
    else
        echo "   ⚠️  Secret ${secret} not found (run store_api_keys.sh)"
    fi
done

echo ""
echo "=================================="
echo "✅ Verification complete!"
echo ""
echo "Next steps:"
echo "1. Connect Dataform to GitHub (UI)"
echo "2. Store API keys: ./scripts/setup/store_api_keys.sh"
echo "3. Deploy Cloud Functions (if using)"
echo "4. Create Cloud Scheduler jobs: ./scripts/deployment/create_cloud_scheduler_jobs.sh"

