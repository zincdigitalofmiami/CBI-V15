#!/bin/bash
# Create Cloud Scheduler jobs for data ingestion and Dataform runs
# Run after API keys are stored and Dataform is connected

set -e

PROJECT_ID="cbi-v15"
REGION="us-central1"
TIMEZONE="America/New_York"

echo "📅 Creating Cloud Scheduler Jobs"
echo "================================"
echo ""

# Service account for Cloud Functions/Cloud Run
SERVICE_ACCOUNT="cbi-v15-functions@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. Databento Daily Ingestion (1 hour)
echo "Creating Databento daily ingestion job..."
gcloud scheduler jobs create http databento-daily-ingestion \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 * * * *" \
    --time-zone="${TIMEZONE}" \
    --uri="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/databento-ingestion" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --description="Hourly Databento price data ingestion" \
    --max-retry-attempts=3 \
    --max-retry-duration=3600s \
    || echo "Job may already exist"

# 2. FRED Daily Ingestion (Daily at 2 AM)
echo "Creating FRED daily ingestion job..."
gcloud scheduler jobs create http fred-daily-ingestion \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 2 * * *" \
    --time-zone="${TIMEZONE}" \
    --uri="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/fred-ingestion" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --description="Daily FRED economic data ingestion" \
    --max-retry-attempts=3 \
    || echo "Job may already exist"

# 3. ScrapeCreators News Ingestion (Every 4 hours)
echo "Creating ScrapeCreators news ingestion job..."
gcloud scheduler jobs create http scrapecreators-news-ingestion \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 */4 * * *" \
    --time-zone="${TIMEZONE}" \
    --uri="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/scrapecreators-ingestion" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --description="4-hourly ScrapeCreators news ingestion" \
    --max-retry-attempts=3 \
    || echo "Job may already exist"

# 4. Dataform Staging Run (Daily at 3 AM)
echo "Creating Dataform staging run job..."
gcloud scheduler jobs create http dataform-staging-run \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 3 * * *" \
    --time-zone="${TIMEZONE}" \
    --uri="https://dataform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/repositories/CBI-V15/workspaces/main:run" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --headers="Content-Type=application/json" \
    --message-body='{"tags":["staging"]}' \
    --description="Daily Dataform staging layer run" \
    --max-retry-attempts=2 \
    || echo "Job may already exist"

# 5. Dataform Features Run (Daily at 4 AM, after staging)
echo "Creating Dataform features run job..."
gcloud scheduler jobs create http dataform-features-run \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 4 * * *" \
    --time-zone="${TIMEZONE}" \
    --uri="https://dataform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/repositories/CBI-V15/workspaces/main:run" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --headers="Content-Type=application/json" \
    --message-body='{"tags":["features"]}' \
    --description="Daily Dataform features layer run" \
    --max-retry-attempts=2 \
    || echo "Job may already exist"

echo ""
echo "✅ Cloud Scheduler jobs created!"
echo ""
echo "Note: Cloud Functions need to be deployed first."
echo "See: scripts/deployment/deploy_cloud_functions.sh"

