#!/bin/bash
# Setup BigQuery Skeleton Structure
# Creates datasets and skeleton tables

set -e

PROJECT_ID="cbi-v15"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🚀 Setting up BigQuery skeleton structure for CBI-V15..."
echo ""

# Step 1: Create datasets
echo "📊 Step 1: Creating BigQuery datasets..."
python3 "$SCRIPT_DIR/create_bigquery_datasets.py"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create datasets"
    exit 1
fi

echo "✅ Datasets created successfully"
echo ""

# Step 2: Create skeleton tables
echo "📋 Step 2: Creating skeleton tables..."
bq query --use_legacy_sql=false --project_id="$PROJECT_ID" < "$SCRIPT_DIR/create_skeleton_tables.sql"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create skeleton tables"
    exit 1
fi

echo "✅ Skeleton tables created successfully"
echo ""

# Step 3: Verify structure
echo "🔍 Step 3: Verifying structure..."
bq ls --project_id="$PROJECT_ID" --format=prettyjson | jq -r '.[] | select(.datasetReference.projectId == "'"$PROJECT_ID"'") | .datasetReference.datasetId' | while read dataset; do
    echo "  📁 Dataset: $dataset"
    bq ls --project_id="$PROJECT_ID" --format=prettyjson "$dataset" | jq -r '.[] | .tableReference.tableId' | while read table; do
        echo "    📄 Table: $table"
    done
done

echo ""
echo "✅ BigQuery skeleton structure setup complete!"
echo ""
echo "Next steps:"
echo "  1. Implement USDA ingestion scripts"
echo "  2. Implement CFTC ingestion scripts"
echo "  3. Implement EIA ingestion scripts"
echo "  4. Build Dataform feature tables"

