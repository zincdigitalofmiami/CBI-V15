#!/bin/bash
# Comprehensive system status check for CBI-V15
# DuckDB/MotherDuck architecture (NO BigQuery, NO Dataform)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "╔════════════════════════════════════════════════╗"
echo "║     CBI-V15 System Status Check                ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# MotherDuck Connection
echo "1️⃣  MotherDuck Connection:"
if [ -n "$MOTHERDUCK_TOKEN" ]; then
    echo "   ✅ MOTHERDUCK_TOKEN set"
    # Test connection
    if python3 -c "import duckdb; duckdb.connect('md:cbi_v15?motherduck_token=$MOTHERDUCK_TOKEN').execute('SELECT 1')" 2>/dev/null; then
        echo "   ✅ Connection successful"
    else
        echo "   ⚠️  Connection failed (check token)"
    fi
else
    echo "   ⚠️  MOTHERDUCK_TOKEN not set"
fi

# Local DuckDB
echo ""
echo "2️⃣  Local DuckDB:"
LOCAL_DB="$PROJECT_ROOT/data/duckdb/cbi_v15.duckdb"
if [ -f "$LOCAL_DB" ]; then
    SIZE=$(du -h "$LOCAL_DB" | cut -f1)
    echo "   ✅ Local database exists ($SIZE)"
else
    echo "   ⚠️  Local database not found"
    echo "   📋 Run: python scripts/setup/execute_local_duckdb_schema.py"
fi

# SQL Macros (AnoFox)
echo ""
echo "3️⃣  SQL Macros (AnoFox):"
if [ -d "$PROJECT_ROOT/database/macros" ]; then
    SQL_COUNT=$(find "$PROJECT_ROOT/database/macros" -name "*.sql" 2>/dev/null | wc -l | tr -d ' ')
    LINES=$(find "$PROJECT_ROOT/database/macros" -name "*.sql" -exec cat {} \; 2>/dev/null | wc -l | tr -d ' ')
    echo "   ✅ Macros directory exists"
    echo "   📊 SQL files: $SQL_COUNT"
    echo "   📊 Total lines: $LINES"
else
    echo "   ❌ Macros directory not found"
fi

# Database Models (DDL)
echo ""
echo "4️⃣  Database Definitions:"
if [ -d "$PROJECT_ROOT/database/models" ]; then
    DEF_COUNT=$(find "$PROJECT_ROOT/database/models" -name "*.sql" 2>/dev/null | wc -l | tr -d ' ')
    echo "   ✅ Models directory exists"
    echo "   📊 SQL definition files: $DEF_COUNT"
else
    echo "   ⚠️  Models directory not found"
fi

# API Keys
echo ""
echo "5️⃣  API Keys (Keychain):"
KEYS=("DATABENTO_API_KEY" "SCRAPECREATORS_API_KEY" "OPENAI_API_KEY" "MOTHERDUCK_TOKEN" "FRED_API_KEY" "EIA_API_KEY")
for key in "${KEYS[@]}"; do
    if security find-generic-password -s "$key" &> /dev/null; then
        echo "   ✅ $key"
    else
        echo "   ⚠️  $key (not in Keychain)"
    fi
done

# Ingestion Scripts
echo ""
echo "6️⃣  Ingestion Scripts:"
if [ -d "$PROJECT_ROOT/trigger" ]; then
    INGEST_PY_COUNT=$(find "$PROJECT_ROOT/trigger" -path "*/Scripts/*.py" 2>/dev/null | wc -l | tr -d ' ')
    INGEST_TS_COUNT=$(find "$PROJECT_ROOT/trigger" -path "*/Scripts/*.ts" 2>/dev/null | wc -l | tr -d ' ')
    echo "   ✅ trigger/ source folders exist"
    echo "   📊 Python scripts: $INGEST_PY_COUNT"
    echo "   📊 TypeScript jobs: $INGEST_TS_COUNT"
else
    echo "   ⚠️  trigger/ not found"
fi

# Training Scripts
echo ""
echo "7️⃣  Training Scripts:"
if [ -d "$PROJECT_ROOT/src/training" ]; then
    TRAIN_COUNT=$(find "$PROJECT_ROOT/src/training" -name "*.py" -not -name "__init__.py" 2>/dev/null | wc -l | tr -d ' ')
    echo "   ✅ Training directory exists"
    echo "   📊 Python scripts: $TRAIN_COUNT"
else
    echo "   ⚠️  Training directory not found"
fi

# Trigger.dev Jobs
echo ""
echo "8️⃣  Trigger.dev Jobs:"
if [ -d "$PROJECT_ROOT/trigger" ]; then
    TRIGGER_COUNT=$(find "$PROJECT_ROOT/trigger" -name "*.ts" 2>/dev/null | wc -l | tr -d ' ')
    echo "   ✅ Trigger directory exists"
    echo "   📊 TypeScript jobs: $TRIGGER_COUNT"
else
    echo "   ⚠️  Trigger directory not found"
fi

# Dashboard
echo ""
echo "9️⃣  Dashboard:"
if [ -d "$PROJECT_ROOT/dashboard" ]; then
    if [ -f "$PROJECT_ROOT/dashboard/package.json" ]; then
        echo "   ✅ Next.js dashboard exists"
    else
        echo "   ⚠️  Dashboard directory exists but no package.json"
    fi
else
    echo "   ⚠️  Dashboard not found"
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Summary                                       ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Architecture: DuckDB/MotherDuck (NO BigQuery, NO Dataform)"
echo ""
echo "📋 Next Steps:"
echo "   1. Ensure MOTHERDUCK_TOKEN is set in .env"
echo "   2. Run: python scripts/setup/execute_local_duckdb_schema.py"
echo "   3. Run: python scripts/setup/deploy_schema_to_motherduck.py"
echo "   4. Begin data ingestion with: python trigger/DataBento/Scripts/collect_daily.py"
