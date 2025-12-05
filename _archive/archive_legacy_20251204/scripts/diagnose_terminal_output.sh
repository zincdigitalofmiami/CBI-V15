#!/bin/bash
# Diagnostic script to test terminal output visibility
# Run this manually: bash scripts/diagnose_terminal_output.sh

echo "=========================================="
echo "TERMINAL OUTPUT DIAGNOSTIC"
echo "=========================================="
echo ""
echo "✅ If you can see this, terminal output IS working"
echo ""
echo "Test 1: Basic output"
echo "  This is stdout"
echo "  This is stderr" >&2
echo ""
echo "Test 2: Environment variables"
echo "  CBI_V15_PROJECT: ${CBI_V15_PROJECT:-NOT_SET}"
echo "  GOOGLE_CLOUD_PROJECT: ${GOOGLE_CLOUD_PROJECT:-NOT_SET}"
echo "  PWD: $(pwd)"
echo ""
echo "Test 3: Command execution"
ls -la .cursorrules 2>&1 | head -3
echo ""
echo "Test 4: BigQuery access"
unset -f bq 2>/dev/null || true
# Temporarily unset PYTHONPATH to avoid credential_loader import conflict
OLD_PYTHONPATH=$PYTHONPATH
unset PYTHONPATH
command bq ls --project_id=cbi-v15 2>&1 | head -5
export PYTHONPATH=$OLD_PYTHONPATH
echo ""
echo "=========================================="
echo "If you see ALL of the above, terminal output is working"
echo "If you DON'T see this, there's a UI/display issue"
echo "=========================================="

