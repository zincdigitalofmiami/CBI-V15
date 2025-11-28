#!/bin/bash
# Grant Dataform service account access to secrets
# Usage: ./grant_dataform_secret_access.sh

set -e

PROJECT_ID="cbi-v15"
PROJECT_NUMBER="287642409540"
DATAFORM_SERVICE_ACCOUNT="service-${PROJECT_NUMBER}@gcp-sa-dataform.iam.gserviceaccount.com"

echo "🔐 Granting Dataform Service Account Secret Access"
echo "==================================================="
echo ""
echo "Project: ${PROJECT_ID}"
echo "Service Account: ${DATAFORM_SERVICE_ACCOUNT}"
echo ""

# List of secrets that Dataform needs access to
SECRETS=(
    "dataform-github-ssh-key"
    "databento-api-key"
    "scrapecreators-api-key"
    "fred-api-key"
    "glide-api-key"
)

echo "Granting access to secrets..."
echo ""

for SECRET_NAME in "${SECRETS[@]}"; do
    echo "📝 Processing: ${SECRET_NAME}"
    
    # Check if secret exists
    if ! gcloud secrets describe "${SECRET_NAME}" --project="${PROJECT_ID}" &> /dev/null; then
        echo "   ⚠️  Secret does not exist, skipping..."
        continue
    fi
    
    # Grant access
    if gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
        --project="${PROJECT_ID}" \
        --member="serviceAccount:${DATAFORM_SERVICE_ACCOUNT}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet 2>&1; then
        echo "   ✅ Access granted"
    else
        echo "   ⚠️  Failed to grant access (may already exist)"
    fi
done

echo ""
echo "✅ Secret access configuration complete!"
echo ""
echo "📋 Verify access:"
echo "  gcloud secrets get-iam-policy dataform-github-ssh-key --project=${PROJECT_ID}"
echo ""
echo "📋 Dataform service account:"
echo "  ${DATAFORM_SERVICE_ACCOUNT}"

