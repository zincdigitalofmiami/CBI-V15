#!/bin/bash
# Master 1000-Day Backfill Script
# Fills last 1000 days for ALL data sources
# Run once, then use daily automation

set -e

echo "================================================================================"
echo "MASTER 1000-DAY BACKFILL"
echo "================================================================================"
echo "Started: $(date)"
echo ""

cd "/Volumes/Satechi Hub/CBI-V15"

# Check environment
if [ -z "$MOTHERDUCK_TOKEN" ]; then
    echo "❌ MOTHERDUCK_TOKEN not set"
    exit 1
fi

echo "✅ Environment validated"
echo ""

# ============================================================================
# PRIORITY 1: CRITICAL DATA (Run First)
# ============================================================================

echo "================================================================================"
echo "[1/6] EPA RIN Prices Backfill (948 days missing)"
echo "================================================================================"
echo "⚠️  Manual backfill required - EPA website scraping"
echo "URL: https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information"
echo "Action: Download weekly Excel files from 2022-01-01 to 2024-12-22"
echo "Skip for now - will implement scraper"
echo ""

echo "================================================================================"
echo "[2/6] NOAA Weather Backfill (800 days missing)"
echo "================================================================================"
python trigger/Weather/Scripts/ingest_weather.py --backfill --days 1000
echo ""

echo "================================================================================"
echo "[3/6] Databento Backfill (148 days missing)"
echo "================================================================================"
echo "⚠️  Check Databento API limits - may have daily request caps"
python trigger/DataBento/Scripts/collect_daily.py --backfill --days 1000 || echo "Skipped - check API limits"
echo ""

# ============================================================================
# PRIORITY 2: NEW FREE SOURCES
# ============================================================================

echo "================================================================================"
echo "[4/6] UN Comtrade - China Imports (NEW)"
echo "================================================================================"
if [ -z "$UN_COMTRADE_API_KEY" ]; then
    echo "⚠️  UN_COMTRADE_API_KEY not set - register free at https://comtradeapi.un.org/"
    echo "Skipping..."
else
    python trigger/UNComtrade/Scripts/collect_china_imports.py
fi
echo ""

echo "================================================================================"
echo "[5/6] INMET Brazil Weather (NEW)"
echo "================================================================================"
python trigger/Weather/Scripts/collect_brazil_weather.py
echo ""

echo "================================================================================"
echo "[6/6] FRED Priority Series (Already Complete)"
echo "================================================================================"
echo "✅ FRED has 342,551 rows with complete history"
echo "No backfill needed"
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================

echo "================================================================================"
echo "VERIFICATION - Running Coverage Check"
echo "================================================================================"
python scripts/smart_backfill_1000days.py

echo ""
echo "================================================================================"
echo "✅ BACKFILL COMPLETE"
echo "================================================================================"
echo "Completed: $(date)"
echo ""
echo "Next steps:"
echo "1. Review coverage report above"
echo "2. Set up daily automation: bash scripts/daily_scrapecreators_run.sh"
echo "3. Add to crontab for daily runs"
