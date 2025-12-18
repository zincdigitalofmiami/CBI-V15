#!/bin/bash
# Quick Test Training Run
# Purpose: Validate training infrastructure with minimal features
# Expected time: 45 minutes

set -e  # Exit on error

echo "================================================================================"
echo "CBI-V15 QUICK TEST TRAINING RUN"
echo "================================================================================"
echo "Started: $(date)"
echo ""

# Check environment
if [ -z "$MOTHERDUCK_TOKEN" ] && [ -n "$motherduck_storage_MOTHERDUCK_TOKEN" ]; then
    export MOTHERDUCK_TOKEN="$motherduck_storage_MOTHERDUCK_TOKEN"
fi

if [ -z "$MOTHERDUCK_TOKEN" ]; then
    echo "❌ ERROR: No MotherDuck token set"
    echo "   Set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN"
    exit 1
fi

echo "✅ Environment validated"
echo ""

# Step 1: Build features
echo "================================================================================"
echo "STEP 1: Building Features (10 minutes)"
echo "================================================================================"
python src/engines/anofox/build_all_features.py

echo ""
echo "✅ Features built"
echo ""

# Step 2: Sync to local
echo "================================================================================"
echo "STEP 2: Syncing to Local DuckDB (5 minutes)"
echo "================================================================================"
python scripts/sync_motherduck_to_local.py

echo ""
echo "✅ Data synced"
echo ""

# Step 3: Train baseline
echo "================================================================================"
echo "STEP 3: Training Baseline Model (30 minutes)"
echo "================================================================================"
python src/training/baselines/lightgbm_zl.py

echo ""
echo "================================================================================"
echo "✅ TEST RUN COMPLETE"
echo "================================================================================"
echo "Completed: $(date)"
echo ""
echo "Next steps:"
echo "1. Review model metrics in output above"
echo "2. Check model artifact: data/models/lightgbm_zl_baseline.pkl"
echo "3. If successful, proceed with EPA RIN backfill"
echo ""


