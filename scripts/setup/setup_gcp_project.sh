#!/bin/bash
# GCP Project Setup for CBI-V15
# Creates GCP project, enables APIs, creates BigQuery datasets

set -e

PROJECT_ID="cbi-v15"
REGION="us-central1"
LOCATION="us-central1"

echo "🚀 Setting up GCP project: $PROJECT_ID"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if project exists
if gcloud projects describe $PROJECT_ID &> /dev/null; then
    echo "✅ Project $PROJECT_ID already exists"
else
    echo "📝 Creating project $PROJECT_ID..."
    gcloud projects create $PROJECT_ID --name="CBI-V15 Soybean Oil Forecasting"
    
    # Set as current project
    gcloud config set project $PROJECT_ID
    
    # Link billing account (user will need to do this manually)
    echo "⚠️  IMPORTANT: Link billing account manually:"
    echo "   gcloud billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID"
    echo "   Or via console: https://console.cloud.google.com/billing"
    read -p "Press Enter after billing is linked..."
fi

# Set as current project
gcloud config set project $PROJECT_ID

# Enable required APIs (quant finance data pipeline)
echo "🔧 Enabling required APIs..."
gcloud services enable \
    bigquery.googleapis.com \
    bigqueryconnection.googleapis.com \
    bigquerymigration.googleapis.com \
    dataform.googleapis.com \
    secretmanager.googleapis.com \
    cloudscheduler.googleapis.com \
    cloudfunctions.googleapis.com \
    run.googleapis.com \
    storage-api.googleapis.com \
    storage-component.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    pubsub.googleapis.com

# Create BigQuery datasets
echo "📊 Creating BigQuery datasets..."

# Quant finance inspired datasets (following GS Quant/JPM patterns)
datasets=(
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

for dataset_info in "${datasets[@]}"; do
    IFS=':' read -r dataset_name description <<< "$dataset_info"
    echo "  Creating dataset: $dataset_name"
    
    if bq show "$PROJECT_ID:$dataset_name" &> /dev/null; then
        echo "    ✅ Dataset $dataset_name already exists"
    else
        bq mk \
            --dataset \
            --location=$LOCATION \
            --description="$description" \
            "$PROJECT_ID:$dataset_name"
        echo "    ✅ Created dataset $dataset_name"
    fi
done

# Create service account for Dataform/Cloud Scheduler
echo "🔐 Creating service accounts..."

SA_NAME="cbi-v15-dataform"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if gcloud iam service-accounts describe $SA_EMAIL &> /dev/null; then
    echo "  ✅ Service account $SA_NAME already exists"
else
    gcloud iam service-accounts create $SA_NAME \
        --display-name="CBI-V15 Dataform Service Account" \
        --description="Service account for Dataform ETL and Cloud Scheduler"
    echo "  ✅ Created service account $SA_NAME"
fi

# Grant permissions
echo "🔑 Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/secretmanager.secretAccessor"

echo ""
echo "✅ GCP setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Store API keys in Secret Manager:"
echo "   gcloud secrets create databento-api-key --data-file=-"
echo ""
echo "2. Store API keys in macOS Keychain (for local scripts):"
echo "   security add-generic-password -a databento -s DATABENTO_API_KEY -w YOUR_KEY"
echo ""
echo "3. Initialize Dataform:"
echo "   cd dataform && npm install && dataform init"
echo ""
echo "4. Verify connections:"
echo "   python scripts/setup/verify_connections.py"

