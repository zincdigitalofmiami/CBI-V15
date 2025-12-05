#!/bin/bash
# IAM Permissions Setup for CBI-V15
# Sets up service accounts and permissions for App Development folder structure

set -e

PROJECT_ID="cbi-v15"
APP_DEV_FOLDER_ID="568609080192"
LOCATION="us-central1"

echo "🔐 Setting up IAM permissions for CBI-V15"
echo "=========================================="
echo ""

# Verify project exists and is in correct folder
echo "1. Verifying project location..."
PROJECT_PARENT=$(gcloud projects describe $PROJECT_ID --format="get(parent)" 2>/dev/null || echo "")
if [[ "$PROJECT_PARENT" != *"$APP_DEV_FOLDER_ID"* ]]; then
    echo "   ⚠️  Project not in App Development folder"
    echo "   Moving project to App Development folder..."
    gcloud projects move $PROJECT_ID --folder=$APP_DEV_FOLDER_ID
    echo "   ✅ Project moved to App Development folder"
else
    echo "   ✅ Project is in App Development folder"
fi

# Set as current project
gcloud config set project $PROJECT_ID

# ============================================================================
# SERVICE ACCOUNTS
# ============================================================================

echo ""
echo "2. Creating service accounts..."

# Service Account 1: Dataform ETL
SA_DATAFORM_NAME="cbi-v15-dataform"
SA_DATAFORM_EMAIL="$SA_DATAFORM_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if gcloud iam service-accounts describe $SA_DATAFORM_EMAIL &> /dev/null; then
    echo "   ✅ Service account $SA_DATAFORM_NAME already exists"
else
    gcloud iam service-accounts create $SA_DATAFORM_NAME \
        --display-name="CBI-V15 Dataform Service Account" \
        --description="Service account for Dataform ETL and Cloud Scheduler"
    echo "   ✅ Created service account $SA_DATAFORM_NAME"
fi

# Service Account 2: Cloud Functions (Ingestion)
SA_FUNCTIONS_NAME="cbi-v15-functions"
SA_FUNCTIONS_EMAIL="$SA_FUNCTIONS_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if gcloud iam service-accounts describe $SA_FUNCTIONS_EMAIL &> /dev/null; then
    echo "   ✅ Service account $SA_FUNCTIONS_NAME already exists"
else
    gcloud iam service-accounts create $SA_FUNCTIONS_NAME \
        --display-name="CBI-V15 Cloud Functions Service Account" \
        --description="Service account for Cloud Functions (data ingestion)"
    echo "   ✅ Created service account $SA_FUNCTIONS_NAME"
fi

# Service Account 3: Cloud Run (Dashboard)
SA_RUN_NAME="cbi-v15-run"
SA_RUN_EMAIL="$SA_RUN_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if gcloud iam service-accounts describe $SA_RUN_EMAIL &> /dev/null; then
    echo "   ✅ Service account $SA_RUN_NAME already exists"
else
    gcloud iam service-accounts create $SA_RUN_NAME \
        --display-name="CBI-V15 Cloud Run Service Account" \
        --description="Service account for Cloud Run (dashboard)"
    echo "   ✅ Created service account $SA_RUN_NAME"
fi

# ============================================================================
# PROJECT-LEVEL IAM PERMISSIONS
# ============================================================================

echo ""
echo "3. Granting project-level permissions..."

# Dataform Service Account Permissions
echo "   Granting permissions to $SA_DATAFORM_NAME..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_DATAFORM_EMAIL" \
    --role="roles/bigquery.dataEditor" \
    --condition=None \
    --quiet || echo "     (may already exist)"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_DATAFORM_EMAIL" \
    --role="roles/bigquery.jobUser" \
    --condition=None \
    --quiet || echo "     (may already exist)"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_DATAFORM_EMAIL" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None \
    --quiet || echo "     (may already exist)"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_DATAFORM_EMAIL" \
    --role="roles/dataform.worker" \
    --condition=None \
    --quiet || echo "     (may already exist)"

# Cloud Functions Service Account Permissions
echo "   Granting permissions to $SA_FUNCTIONS_NAME..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_FUNCTIONS_EMAIL" \
    --role="roles/bigquery.dataEditor" \
    --condition=None \
    --quiet || echo "     (may already exist)"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_FUNCTIONS_EMAIL" \
    --role="roles/bigquery.jobUser" \
    --condition=None \
    --quiet || echo "     (may already exist)"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_FUNCTIONS_EMAIL" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None \
    --quiet || echo "     (may already exist)"

# Cloud Run Service Account Permissions
echo "   Granting permissions to $SA_RUN_NAME..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_RUN_EMAIL" \
    --role="roles/bigquery.dataViewer" \
    --condition=None \
    --quiet || echo "     (may already exist)"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_RUN_EMAIL" \
    --role="roles/bigquery.jobUser" \
    --condition=None \
    --quiet || echo "     (may already exist)"

# ============================================================================
# FOLDER-LEVEL IAM PERMISSIONS (App Development Folder)
# ============================================================================

echo ""
echo "4. Checking folder-level permissions..."

# Check if user has folder admin access
CURRENT_USER=$(gcloud config get-value account)
echo "   Current user: $CURRENT_USER"

# Note: Folder-level permissions are typically managed by org admins
# This script documents what permissions are needed but doesn't grant them
echo "   ⚠️  Folder-level permissions must be granted by organization admin"
echo "   Required folder permissions:"
echo "     - roles/resourcemanager.folderViewer (for project visibility)"
echo "     - roles/bigquery.admin (if folder-level BigQuery access needed)"

# ============================================================================
# DATASET-LEVEL PERMISSIONS (BigQuery)
# ============================================================================

echo ""
echo "5. Setting up BigQuery dataset permissions..."

# Grant Dataform SA access to all datasets
DATASETS=("raw" "staging" "features" "training" "forecasts" "api" "reference" "ops")

for dataset in "${DATASETS[@]}"; do
    echo "   Granting access to dataset: $dataset"
    
    # Grant Dataform SA
    bq show --format=prettyjson "$PROJECT_ID:$dataset" &> /dev/null && \
    bq add-iam-policy-binding \
        --member="serviceAccount:$SA_DATAFORM_EMAIL" \
        --role="roles/bigquery.dataEditor" \
        "$PROJECT_ID:$dataset" 2>/dev/null || echo "     (dataset may not exist yet or permission already granted)"
    
    # Grant Cloud Functions SA
    bq show --format=prettyjson "$PROJECT_ID:$dataset" &> /dev/null && \
    bq add-iam-policy-binding \
        --member="serviceAccount:$SA_FUNCTIONS_EMAIL" \
        --role="roles/bigquery.dataEditor" \
        "$PROJECT_ID:$dataset" 2>/dev/null || echo "     (dataset may not exist yet or permission already granted)"
    
    # Grant Cloud Run SA (read-only for api dataset)
    if [ "$dataset" == "api" ]; then
        bq show --format=prettyjson "$PROJECT_ID:$dataset" &> /dev/null && \
        bq add-iam-policy-binding \
            --member="serviceAccount:$SA_RUN_EMAIL" \
            --role="roles/bigquery.dataViewer" \
            "$PROJECT_ID:$dataset" 2>/dev/null || echo "     (dataset may not exist yet or permission already granted)"
    fi
done

# ============================================================================
# CLOUD SCHEDULER PERMISSIONS
# ============================================================================

echo ""
echo "6. Setting up Cloud Scheduler permissions..."

# Grant Cloud Scheduler permission to invoke Cloud Functions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_DATAFORM_EMAIL" \
    --role="roles/cloudfunctions.invoker" \
    --condition=None \
    --quiet || echo "   (may already exist)"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=========================================="
echo "✅ IAM Setup Complete"
echo "=========================================="
echo ""
echo "Service Accounts Created:"
echo "  ✅ $SA_DATAFORM_EMAIL (Dataform ETL)"
echo "  ✅ $SA_FUNCTIONS_EMAIL (Cloud Functions)"
echo "  ✅ $SA_RUN_EMAIL (Cloud Run)"
echo ""
echo "Permissions Granted:"
echo "  ✅ BigQuery Data Editor (Dataform, Functions)"
echo "  ✅ BigQuery Job User (all service accounts)"
echo "  ✅ Secret Manager Accessor (Dataform, Functions)"
echo "  ✅ Dataform Worker (Dataform)"
echo "  ✅ BigQuery Data Viewer (Cloud Run - read-only)"
echo ""
echo "⚠️  Note: Folder-level permissions may require org admin approval"
echo ""
echo "Next Steps:"
echo "  1. Verify folder-level permissions with org admin"
echo "  2. Test service account access"
echo "  3. Configure Cloud Scheduler jobs with service accounts"
echo ""

