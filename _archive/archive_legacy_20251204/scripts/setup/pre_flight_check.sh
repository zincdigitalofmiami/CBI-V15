#!/bin/bash
# Pre-Flight Check for BigQuery Setup
# Verifies all prerequisites before executing BigQuery setup

set -e

PROJECT_ID="cbi-v15"
LOCATION="us-central1"

echo "🔍 Pre-Flight Check for BigQuery Setup"
echo "======================================"
echo ""

# Check 1: Python 3
echo "1. Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "   ❌ Python 3 not found"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "   ✅ Python 3 found: $PYTHON_VERSION"

# Check 2: BigQuery CLI
echo ""
echo "2. Checking BigQuery CLI (bq)..."
if ! command -v bq &> /dev/null; then
    echo "   ❌ BigQuery CLI (bq) not found"
    echo "   Install: https://cloud.google.com/bigquery/docs/bq-command-line-tool"
    exit 1
fi
BQ_VERSION=$(bq version 2>&1 | head -1)
echo "   ✅ BigQuery CLI found: $BQ_VERSION"

# Check 3: Google Cloud SDK
echo ""
echo "3. Checking Google Cloud SDK (gcloud)..."
if ! command -v gcloud &> /dev/null; then
    echo "   ❌ Google Cloud SDK (gcloud) not found"
    echo "   Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
GCLOUD_VERSION=$(gcloud --version 2>&1 | head -1)
echo "   ✅ Google Cloud SDK found: $GCLOUD_VERSION"

# Check 4: GCP Project
echo ""
echo "4. Checking GCP Project..."
CURRENT_PROJECT=$(gcloud config get-value project 2>&1)
if [ -z "$CURRENT_PROJECT" ] || [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo "   ⚠️  Current project: $CURRENT_PROJECT"
    echo "   ⚠️  Expected project: $PROJECT_ID"
    echo "   Setting project to $PROJECT_ID..."
    gcloud config set project $PROJECT_ID
    echo "   ✅ Project set to $PROJECT_ID"
else
    echo "   ✅ Project is set to $PROJECT_ID"
fi

# Check 5: Authentication
echo ""
echo "5. Checking GCP Authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "   ❌ No active GCP authentication found"
    echo "   Run: gcloud auth login"
    exit 1
fi
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)
echo "   ✅ Authenticated as: $ACTIVE_ACCOUNT"

# Check 6: BigQuery API Enabled
echo ""
echo "6. Checking BigQuery API..."
if ! gcloud services list --enabled --filter="name:bigquery.googleapis.com" --format="value(name)" | grep -q bigquery; then
    echo "   ⚠️  BigQuery API not enabled"
    echo "   Enabling BigQuery API..."
    gcloud services enable bigquery.googleapis.com --project=$PROJECT_ID
    echo "   ✅ BigQuery API enabled"
else
    echo "   ✅ BigQuery API is enabled"
fi

# Check 7: Python Dependencies
echo ""
echo "7. Checking Python Dependencies..."
if ! python3 -c "import google.cloud.bigquery" 2>/dev/null; then
    echo "   ⚠️  google-cloud-bigquery not installed"
    echo "   Installing..."
    pip3 install google-cloud-bigquery
    echo "   ✅ google-cloud-bigquery installed"
else
    echo "   ✅ google-cloud-bigquery is installed"
fi

# Check 8: Script Files Exist
echo ""
echo "8. Checking Setup Scripts..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -f "$SCRIPT_DIR/create_bigquery_datasets.py" ]; then
    echo "   ❌ create_bigquery_datasets.py not found"
    exit 1
fi
echo "   ✅ create_bigquery_datasets.py found"

if [ ! -f "$SCRIPT_DIR/create_skeleton_tables.sql" ]; then
    echo "   ❌ create_skeleton_tables.sql not found"
    exit 1
fi
echo "   ✅ create_skeleton_tables.sql found"

if [ ! -f "$SCRIPT_DIR/initialize_reference_tables.sql" ]; then
    echo "   ❌ initialize_reference_tables.sql not found"
    exit 1
fi
echo "   ✅ initialize_reference_tables.sql found"

if [ ! -f "$SCRIPT_DIR/verify_bigquery_setup.py" ]; then
    echo "   ❌ verify_bigquery_setup.py not found"
    exit 1
fi
echo "   ✅ verify_bigquery_setup.py found"

# Check 9: Permissions
echo ""
echo "9. Checking BigQuery Permissions..."
if ! bq ls --project_id=$PROJECT_ID &>/dev/null; then
    echo "   ❌ Cannot access BigQuery (permission denied)"
    echo "   Ensure you have BigQuery Admin or Editor role"
    exit 1
fi
echo "   ✅ BigQuery access confirmed"

# Summary
echo ""
echo "======================================"
echo "✅ Pre-Flight Check Complete"
echo "======================================"
echo ""
echo "All prerequisites verified:"
echo "  ✅ Python 3"
echo "  ✅ BigQuery CLI"
echo "  ✅ Google Cloud SDK"
echo "  ✅ GCP Project: $PROJECT_ID"
echo "  ✅ Authentication"
echo "  ✅ BigQuery API"
echo "  ✅ Python Dependencies"
echo "  ✅ Setup Scripts"
echo "  ✅ Permissions"
echo ""
echo "Ready to execute BigQuery setup!"
echo "Run: ./scripts/setup/setup_bigquery_skeleton.sh"
echo ""

