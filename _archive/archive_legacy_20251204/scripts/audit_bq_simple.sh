#!/bin/bash
# Simple BigQuery audit for CBI-V15
# Shows datasets and tables

# Don't exit on error - handle gracefully
set +e

PROJECT_ID="cbi-v15"

echo "🔍 BigQuery Audit - CBI-V15"
echo "============================"
echo ""
echo "Project: $PROJECT_ID"
echo ""

# List all datasets (unset bq wrapper if exists, unset PYTHONPATH to avoid import conflict)
unset -f bq 2>/dev/null || true
OLD_PYTHONPATH=$PYTHONPATH
unset PYTHONPATH
echo "📊 Datasets:"
echo "------------"
command bq ls --project_id=$PROJECT_ID 2>/dev/null | tail -n +3 | awk '{print "  " $1}' | grep -v '^  $'

echo ""

# For each dataset, list tables (PYTHONPATH already unset above)
DATASETS=$(command bq ls --project_id=$PROJECT_ID 2>/dev/null | tail -n +3 | awk '{print $1}' | grep -v '^$')

if [ -z "$DATASETS" ]; then
    echo "📁 No datasets found"
else
    for dataset in $DATASETS; do
        echo "📁 Dataset: $dataset"
        TABLES_OUTPUT=$(command bq ls --project_id=$PROJECT_ID ${PROJECT_ID}:${dataset} 2>&1)
        TABLES=$(echo "$TABLES_OUTPUT" | tail -n +3 | awk '{print $1}' | grep -v '^$' | grep -v '^datasetId$' || true)
        
        if [ -z "$TABLES" ] || [ "$TABLES" = "" ]; then
            echo "   (no tables)"
        else
            for table in $TABLES; do
                echo "   - $table"
            done
        fi
        
        echo ""
    done
fi

# Restore PYTHONPATH
export PYTHONPATH=$OLD_PYTHONPATH

echo "============================"
echo "✅ Audit complete"





