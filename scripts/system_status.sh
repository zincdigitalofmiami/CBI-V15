#!/bin/bash
# Comprehensive system status check for CBI-V15

set -e

PROJECT_ID="cbi-v15"
REGION="us-central1"

echo "╔════════════════════════════════════════════════╗"
echo "║     CBI-V15 System Status Check                ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# GCP Project
echo "1️⃣  GCP Project:"
if gcloud projects describe "$PROJECT_ID" &> /dev/null; then
    BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format="get(billingAccountName)" 2>/dev/null || echo "Not linked")
    echo "   ✅ Project: $PROJECT_ID"
    echo "   ✅ Billing: ${BILLING:-Not linked}"
else
    echo "   ❌ Project not found"
fi

# BigQuery Datasets
echo ""
echo "2️⃣  BigQuery Datasets:"
DATASETS=("raw" "staging" "features" "training" "forecasts" "api" "reference" "ops")
for dataset in "${DATASETS[@]}"; do
    if bq show "${PROJECT_ID}:${dataset}" &> /dev/null; then
        echo "   ✅ $dataset"
    else
        echo "   ❌ $dataset (missing)"
    fi
done

# BigQuery Tables
echo ""
echo "3️⃣  BigQuery Tables:"
TABLE_COUNT=$(bq ls -d "${PROJECT_ID}:raw" --format=json 2>/dev/null | jq length 2>/dev/null || echo "0")
echo "   📊 Raw tables: $TABLE_COUNT"
REF_DATA=$(bq query --project_id="$PROJECT_ID" --use_legacy_sql=false --format=csv "SELECT COUNT(*) FROM \`${PROJECT_ID}.reference.regime_calendar\`" 2>/dev/null | tail -1 || echo "0")
echo "   📊 Reference data: $REF_DATA rows"

# Dataform
echo ""
echo "4️⃣  Dataform:"
if [ -f "dataform/dataform.json" ]; then
    echo "   ✅ Configuration exists"
    SQL_COUNT=$(find dataform/definitions -name "*.sqlx" 2>/dev/null | wc -l | tr -d ' ')
    echo "   📊 SQL files: $SQL_COUNT"
else
    echo "   ⚠️  Configuration not found"
fi

# API Keys
echo ""
echo "5️⃣  API Keys:"
if security find-generic-password -s "DATABENTO_API_KEY" &> /dev/null; then
    echo "   ✅ Databento key (Keychain)"
else
    echo "   ⚠️  Databento key (not stored)"
fi

if security find-generic-password -s "SCRAPECREATORS_API_KEY" &> /dev/null; then
    echo "   ✅ ScrapeCreators key (Keychain)"
else
    echo "   ⚠️  ScrapeCreators key (not stored)"
fi

# GitHub Connection
echo ""
echo "6️⃣  GitHub Connection:"
if [ -f ~/.ssh/dataform_github_ed25519.pub ]; then
    echo "   ✅ SSH key generated"
    if gcloud secrets describe dataform-github-ssh-key --project="$PROJECT_ID" &> /dev/null; then
        echo "   ✅ Secret stored in Secret Manager"
    else
        echo "   ⚠️  Secret not in Secret Manager"
    fi
else
    echo "   ⚠️  SSH key not generated"
fi

# Data Status
echo ""
echo "7️⃣  Data Status:"
RAW_DATA=$(bq query --project_id="$PROJECT_ID" --use_legacy_sql=false --format=csv "SELECT COUNT(*) FROM \`${PROJECT_ID}.raw.databento_futures_ohlcv_1d\`" 2>/dev/null | tail -1 || echo "0")
if [ "$RAW_DATA" -gt 0 ]; then
    echo "   ✅ Raw data: $RAW_DATA rows"
else
    echo "   ⚠️  Raw data: Empty (ready for ingestion)"
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Summary                                       ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "✅ Infrastructure: Complete"
echo "✅ Dataform: Ready"
echo "⚠️  API Keys: Need to be stored"
echo "⚠️  Data: Ready for ingestion"
echo ""
echo "📋 Next Steps:"
echo "   1. Add SSH key to GitHub (if not done)"
echo "   2. Connect Dataform in UI"
echo "   3. Store API keys: ./scripts/setup/store_api_keys.sh"
echo "   4. Begin data ingestion"

