#!/bin/bash
# ============================================================================
# CBI-V15 Complete System Deployment
# ============================================================================
# This script deploys the entire CBI-V15 system:
# 1. Database schemas and tables (both MotherDuck and local)
# 2. CFTC COT data ingestion
# 3. Feature engineering pipeline
# 4. Verification and validation
#
# Usage:
#   ./scripts/deploy_complete_system.sh
#   ./scripts/deploy_complete_system.sh --skip-cot  # Skip COT ingestion
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_COT=false
if [[ "$1" == "--skip-cot" ]]; then
    SKIP_COT=true
fi

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}CBI-V15 COMPLETE SYSTEM DEPLOYMENT${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# ============================================================================
# Step 1: Environment Check
# ============================================================================
echo -e "${YELLOW}Step 1: Checking environment...${NC}"

if [ -z "$MOTHERDUCK_TOKEN" ]; then
    echo -e "${RED}❌ MOTHERDUCK_TOKEN not set${NC}"
    exit 1
fi

if [ -z "$DATABENTO_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  DATABENTO_API_KEY not set (required for data ingestion)${NC}"
fi

echo -e "${GREEN}✅ Environment check passed${NC}"
echo ""

# ============================================================================
# Step 2: Database Setup (MotherDuck + Local)
# ============================================================================
echo -e "${YELLOW}Step 2: Setting up databases...${NC}"
echo -e "${BLUE}This will create schemas, tables, macros, and views${NC}"
echo ""

python scripts/setup_database.py --both --force

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Database setup failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Database setup complete${NC}"
echo ""

# ============================================================================
# Step 3: CFTC COT Data Ingestion
# ============================================================================
if [ "$SKIP_COT" = false ]; then
    echo -e "${YELLOW}Step 3: Ingesting CFTC COT data...${NC}"
    echo -e "${BLUE}This will download weekly COT reports (2020-present)${NC}"
    echo -e "${BLUE}For full backfill (2006-present), run: python src/ingestion/cftc/ingest_cot.py --backfill${NC}"
    echo ""

    python src/ingestion/cftc/ingest_cot.py --start-year 2020

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ CFTC COT ingestion failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ CFTC COT data ingested${NC}"
    echo ""
else
    echo -e "${YELLOW}Step 3: Skipping CFTC COT ingestion (--skip-cot flag)${NC}"
    echo ""
fi

# ============================================================================
# Step 4: Feature Engineering
# ============================================================================
echo -e "${YELLOW}Step 4: Building features...${NC}"
echo -e "${BLUE}This will build technical indicators and Big 8 bucket features${NC}"
echo ""

python src/engines/anofox/build_all_features.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Feature engineering failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Features built${NC}"
echo ""

# ============================================================================
# Step 5: Verification
# ============================================================================
echo -e "${YELLOW}Step 5: Verifying deployment...${NC}"
echo ""

python scripts/verify_pipeline.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Verification failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Verification passed${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${GREEN}Databases:${NC}"
echo -e "  ✅ MotherDuck (md:cbi-v15)"
echo -e "  ✅ Local DuckDB (data/duckdb/cbi_v15.duckdb)"
echo ""
echo -e "${GREEN}Schemas:${NC}"
echo -e "  ✅ raw (5 tables + CFTC COT)"
echo -e "  ✅ staging (4 tables)"
echo -e "  ✅ features (2 tables)"
echo -e "  ✅ training (1 table)"
echo ""
echo -e "${GREEN}Data:${NC}"
if [ "$SKIP_COT" = false ]; then
    echo -e "  ✅ CFTC COT (2020-present)"
else
    echo -e "  ⚠️  CFTC COT (skipped)"
fi
echo ""
echo -e "${GREEN}Features:${NC}"
echo -e "  ✅ Technical indicators (33 symbols)"
echo -e "  ✅ Big 8 bucket features (with COT)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Ingest market data: python src/ingestion/databento/ingest_daily.py"
echo -e "  2. Ingest FRED data: python src/ingestion/fred/ingest_macro.py"
echo -e "  3. Ingest EIA data: python src/ingestion/eia/ingest_biofuels.py"
echo -e "  4. Run full COT backfill: python src/ingestion/cftc/ingest_cot.py --backfill"
echo ""

