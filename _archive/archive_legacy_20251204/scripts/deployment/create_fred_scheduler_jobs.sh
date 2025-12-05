#!/bin/bash
# Create Cloud Scheduler jobs for FRED ingestion (bucketed by role)
# Each job triggers a separate Cloud Function / Cloud Run endpoint
# responsible ONLY for its own FRED bucket.
#
# IMPORTANT:
# - These jobs are cheap (Cloud Scheduler only).
# - Functions/services they call must be deployed separately.
# - Names/URIs should match your deploy script (deploy_cloud_functions.sh).

set -e

PROJECT_ID="cbi-v15"
REGION="us-central1"
TIMEZONE="America/New_York"

SERVICE_ACCOUNT="cbi-v15-functions@${PROJECT_ID}.iam.gserviceaccount.com"

echo "📅 Creating FRED Cloud Scheduler Jobs (bucketed)"
echo "==============================================="
echo ""

# 1. FRED FX (BRL, DXY, other FX indices)
echo "Creating FRED FX ingestion job..."
gcloud scheduler jobs create http fred-fx-daily \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 2 * * *" \
    --time-zone="${TIMEZONE}" \
    --uri="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/fred-fx-ingestion" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --description="Daily FRED FX spot/index ingestion (BRL, DXY, etc.)" \
    --max-retry-attempts=3 \
    --max-retry-duration=1800s \
    || echo "Job fred-fx-daily may already exist"

# 2. FRED Rates & Yield Curve (policy + term structure)
echo "Creating FRED rates/curve ingestion job..."
gcloud scheduler jobs create http fred-rates-curve-daily \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="5 2 * * *" \
    --time-zone="${TIMEZONE}" \
    --uri="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/fred-rates-curve-ingestion" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --description="Daily FRED policy rates and yield curve ingestion" \
    --max-retry-attempts=3 \
    --max-retry-duration=1800s \
    || echo "Job fred-rates-curve-daily may already exist"

# 3. FRED Financial Conditions (NFCI, leverage, etc.)
echo "Creating FRED financial-conditions ingestion job..."
gcloud scheduler jobs create http fred-financial-conditions-daily \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="10 2 * * *" \
    --time-zone="${TIMEZONE}" \
    --uri="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/fred-financial-conditions-ingestion" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --description="Daily FRED financial conditions ingestion (NFCI, NFCILEVERAGE, etc.)" \
    --max-retry-attempts=3 \
    --max-retry-duration=1800s \
    || echo "Job fred-financial-conditions-daily may already exist"

echo ""
echo "✅ FRED Cloud Scheduler jobs created (FX, rates/curve, financial conditions)."
echo "Note: Ensure matching Cloud Functions/Run services are deployed:"
echo "  - fred-fx-ingestion"
echo "  - fred-rates-curve-ingestion"
echo "  - fred-financial-conditions-ingestion"






