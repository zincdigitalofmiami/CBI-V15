#!/bin/bash
# Verify AI assistant can access required resources
# Checks: environment variables, gcloud/bq, GCP auth, BigQuery, Python, external drive

set -e

echo "🔍 AI Access Verification"
echo "========================"
echo ""

# Check environment variables
echo "📋 Environment Variables:"
if [ -n "$CBI_V15_PROJECT" ]; then
    echo "  ✅ CBI_V15_PROJECT: $CBI_V15_PROJECT"
else
    echo "  ❌ CBI_V15_PROJECT: Not set"
fi

if [ -n "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "  ✅ GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
else
    echo "  ❌ GOOGLE_CLOUD_PROJECT: Not set"
fi

if [ -n "$PYTHONPATH" ]; then
    echo "  ✅ PYTHONPATH: $PYTHONPATH"
else
    echo "  ⚠️  PYTHONPATH: Not set (may be OK)"
fi

echo ""

# Check gcloud
echo "☁️  GCP Tools:"
if command -v gcloud &> /dev/null; then
    echo "  ✅ gcloud: $(which gcloud)"
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "not set")
    echo "  📌 Current project: $CURRENT_PROJECT"
    
    # Check authentication
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
        ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
        echo "  ✅ Authenticated as: $ACTIVE_ACCOUNT"
    else
        echo "  ❌ Not authenticated - run: gcloud auth login"
    fi
else
    echo "  ❌ gcloud: Not found in PATH"
fi

echo ""

# Check bq
if command -v bq &> /dev/null; then
    echo "  ✅ bq: $(which bq)"
    # Test bq access (unset function wrapper if exists, unset PYTHONPATH to avoid import conflict)
    unset -f bq 2>/dev/null || true
    OLD_PYTHONPATH=$PYTHONPATH
    unset PYTHONPATH
    if command bq ls --project_id=cbi-v15 2>&1 | head -1 | grep -q "datasetId"; then
        echo "  ✅ BigQuery access: Working"
        DATASET_COUNT=$(command bq ls --project_id=cbi-v15 --format=csv 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
        echo "  📊 Datasets found: $DATASET_COUNT"
    else
        echo "  ❌ BigQuery access: Failed"
        echo "  Debug: $(command bq ls --project_id=cbi-v15 2>&1 | head -3)"
    fi
    export PYTHONPATH=$OLD_PYTHONPATH
else
    echo "  ❌ bq: Not found in PATH"
fi

echo ""

# Check Python
echo "🐍 Python Environment:"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "  ✅ python3: $PYTHON_VERSION"
    PYTHON_PATH=$(which python3)
    echo "  📍 Path: $PYTHON_PATH"
    
    # Check if keychain_manager can be imported
    if python3 -c "import sys; sys.path.insert(0, 'src'); from cbi_utils.keychain_manager import get_api_key; print('✅ keychain_manager imports OK')" 2>/dev/null; then
        echo "  ✅ keychain_manager: Can import"
    else
        echo "  ❌ keychain_manager: Import failed"
    fi
else
    echo "  ❌ python3: Not found"
fi

echo ""

# Check external drive
echo "💾 External Drive Access:"
if [ -d "/Volumes/Satechi Hub" ]; then
    echo "  ✅ External drive mounted: /Volumes/Satechi Hub"
    DRIVE_SIZE=$(df -h "/Volumes/Satechi Hub" 2>/dev/null | tail -1 | awk '{print $4}')
    echo "  📊 Available space: $DRIVE_SIZE"
    
    # Check if backup exists
    if [ -d "/Volumes/Satechi Hub/CBI-V14-Backup-2025-11-28" ]; then
        BACKUP_SIZE=$(du -sh "/Volumes/Satechi Hub/CBI-V14-Backup-2025-11-28" 2>/dev/null | cut -f1)
        echo "  ✅ V14 backup found: $BACKUP_SIZE"
    else
        echo "  ⚠️  V14 backup not found"
    fi
else
    echo "  ⚠️  External drive not mounted at /Volumes/Satechi Hub"
fi

echo ""

# Summary
echo "========================"
echo "✅ Verification complete"
echo ""
echo "If all checks show ✅, AI assistants should be able to access:"
echo "  - BigQuery (via bq CLI)"
echo "  - GCP resources (via gcloud CLI)"
echo "  - External drive (via file system)"
echo "  - Python modules (via PYTHONPATH)"





