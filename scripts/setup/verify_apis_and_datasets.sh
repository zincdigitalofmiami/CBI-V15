#!/bin/bash
# Verify all required APIs and datasets for quant finance architecture
# Based on GS Quant, JPM, and industry best practices

set -e

PROJECT_ID="cbi-v15"

echo "🔍 Verifying APIs and Datasets for Quant Finance Architecture"
echo "============================================================"
echo ""

# Required APIs for quant finance data pipeline
REQUIRED_APIS=(
    "bigquery.googleapis.com:BigQuery Data Warehouse"
    "bigqueryconnection.googleapis.com:BigQuery Connections (federated queries)"
    "bigquerymigration.googleapis.com:BigQuery Migration (optional but useful)"
    "dataform.googleapis.com:Dataform ETL Framework (CRITICAL)"
    "secretmanager.googleapis.com:Secret Manager (API keys)"
    "cloudscheduler.googleapis.com:Cloud Scheduler (daily jobs)"
    "cloudfunctions.googleapis.com:Cloud Functions (serverless ingestion)"
    "run.googleapis.com:Cloud Run (containerized jobs)"
    "storage-api.googleapis.com:Cloud Storage API"
    "storage-component.googleapis.com:Cloud Storage Component"
    "logging.googleapis.com:Cloud Logging (monitoring)"
    "monitoring.googleapis.com:Cloud Monitoring (metrics)"
    "pubsub.googleapis.com:Pub/Sub (event-driven ingestion, optional)"
)

# Quant finance inspired datasets (following GS Quant/JPM patterns)
REQUIRED_DATASETS=(
    "raw:Raw source data (market, economic, weather, news)"
    "staging:Cleaned normalized data (point-in-time discipline)"
    "features:Engineered features (Big 8 drivers, technical indicators)"
    "training:Training-ready tables (with targets and regime weights)"
    "forecasts:Model predictions (multi-horizon forecasts)"
    "signals:Trading signals and derived indicators"
    "reference:Reference data (calendars, symbols, mappings)"
    "api:Public API views (dashboard-ready)"
    "ops:Operations monitoring (data quality, model performance)"
)

echo "📋 Checking Required APIs..."
echo ""

MISSING_APIS=()
for api_info in "${REQUIRED_APIS[@]}"; do
    IFS=':' read -r api_name api_desc <<< "$api_info"
    if gcloud services list --enabled --project="$PROJECT_ID" --filter="name:$api_name" --format="value(name)" | grep -q "$api_name"; then
        echo "  ✅ $api_name - $api_desc"
    else
        echo "  ❌ $api_name - $api_desc (MISSING)"
        MISSING_APIS+=("$api_name")
    fi
done

echo ""
echo "📊 Checking Required Datasets..."
echo ""

MISSING_DATASETS=()
for dataset_info in "${REQUIRED_DATASETS[@]}"; do
    IFS=':' read -r dataset_name dataset_desc <<< "$dataset_info"
    if bq show "$PROJECT_ID:$dataset_name" &> /dev/null; then
        echo "  ✅ $dataset_name - $dataset_desc"
    else
        echo "  ❌ $dataset_name - $dataset_desc (MISSING)"
        MISSING_DATASETS+=("$dataset_name")
    fi
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"

if [ ${#MISSING_APIS[@]} -eq 0 ] && [ ${#MISSING_DATASETS[@]} -eq 0 ]; then
    echo "✅ All APIs and datasets are configured!"
    exit 0
else
    if [ ${#MISSING_APIS[@]} -gt 0 ]; then
        echo "❌ Missing APIs:"
        for api in "${MISSING_APIS[@]}"; do
            echo "   - $api"
        done
        echo ""
        echo "Enable missing APIs:"
        echo "gcloud services enable ${MISSING_APIS[*]} --project=$PROJECT_ID"
    fi
    
    if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
        echo "❌ Missing Datasets:"
        for dataset in "${MISSING_DATASETS[@]}"; do
            echo "   - $dataset"
        done
        echo ""
        echo "Create missing datasets:"
        for dataset in "${MISSING_DATASETS[@]}"; do
            echo "bq mk --dataset --location=us-central1 --description=\"...\" $PROJECT_ID:$dataset"
        done
    fi
    
    exit 1
fi

