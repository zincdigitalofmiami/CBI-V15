#!/bin/bash
# Setup BigQuery Skeleton for CBI-V15
# Complete setup: Datasets → Tables → Reference Data → Verification

set -e

PROJECT_ID="cbi-v15"
LOCATION="us-central1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🚀 Starting BigQuery skeleton setup for project: $PROJECT_ID in location: $LOCATION"
echo ""

# Step 1: Create BigQuery Datasets
echo "📊 Step 1: Creating BigQuery datasets..."
cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/create_bigquery_datasets.py"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create datasets"
    exit 1
fi

echo "✅ Datasets created successfully"
echo ""

# Step 2: Create Skeleton Tables
echo "📋 Step 2: Creating skeleton tables with partitioning and clustering..."
bq query --project_id="$PROJECT_ID" --location="$LOCATION" --use_legacy_sql=false < "$SCRIPT_DIR/create_skeleton_tables.sql"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create skeleton tables"
    exit 1
fi

echo "✅ Skeleton tables created successfully"
echo ""

# Step 3: Initialize Reference Tables
echo "📚 Step 3: Initializing reference tables (regime calendar, splits, neural drivers)..."
bq query --project_id="$PROJECT_ID" --location="$LOCATION" --use_legacy_sql=false < "$SCRIPT_DIR/initialize_reference_tables.sql"

if [ $? -ne 0 ]; then
    echo "❌ Failed to initialize reference tables"
    exit 1
fi

echo "✅ Reference tables initialized successfully"
echo ""

# Step 4: Verify Structure
echo "🔍 Step 4: Verifying BigQuery setup..."
python3 "$SCRIPT_DIR/verify_bigquery_setup.py"

if [ $? -ne 0 ]; then
    echo "❌ Verification failed"
    exit 1
fi

echo ""
echo "✅ BigQuery skeleton setup complete!"
echo ""
echo "📋 Next Steps:"
echo "  1. Test data ingestion (run one ingestion script)"
echo "  2. Test Dataform compilation (cd dataform && dataform compile)"
echo "  3. Build first feature table (staging.market_daily)"
echo "  4. Validate with Pandera (run validation schema)"
echo ""
