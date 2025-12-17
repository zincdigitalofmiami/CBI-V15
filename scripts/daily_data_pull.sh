#!/bin/bash
# Daily Data Pull - NO TRIGGER, NO CLOUD SERVICES
# Just runs Python scripts directly with API calls
# All APIs are FREE except Databento (~$50/month)

set -e

cd "/Volumes/Satechi Hub/CBI-V15"

echo "================================================================================"
echo "DAILY DATA PULL - $(date)"
echo "================================================================================"

# 1. Databento (53 futures symbols)
echo "[1/4] Databento..."
python src/ingestion/databento/collect_daily.py

# 2. FRED (60+ macro series) 
echo "[2/4] FRED..."
python trigger/FRED/Scripts/collect_fred_priority_series.py

# 3. ScrapeCreators (news/sentiment)
echo "[3/4] ScrapeCreators..."
python trigger/ScrapeCreators/Scripts/collect_all_scrapecreators.py

# 4. Weather (US stations)
echo "[4/4] Weather..."
python trigger/Weather/Scripts/pull_weather_now.py

echo ""
echo "✅ DAILY PULL COMPLETE - $(date)"


