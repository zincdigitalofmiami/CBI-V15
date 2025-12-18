#!/bin/bash
# Daily ScrapeCreators Data Collection
# Run this daily via cron or manually
# Collects ALL Big 8 bucket data from ScrapeCreators

set -e

echo "================================================================================"
echo "DAILY SCRAPECREATORS COLLECTION"
echo "================================================================================"
echo "Started: $(date)"
echo ""

cd "/Volumes/Satechi Hub/CBI-V15"

# Check environment
if [ -z "$SCRAPECREATORS_API_KEY" ]; then
    echo "❌ ERROR: SCRAPECREATORS_API_KEY not set"
    exit 1
fi

if [ -z "$MOTHERDUCK_TOKEN" ]; then
    echo "❌ ERROR: MOTHERDUCK_TOKEN not set"
    exit 1
fi

echo "✅ Environment validated"
echo ""

# Run Big 8 bucket collector (MAIN SCRIPT)
echo "================================================================================"
echo "Collecting Big 8 Bucket Data (Twitter + Google + Reddit)"
echo "================================================================================"
python src/ingestion/scrapecreators/collect_by_big8_buckets.py

echo ""
echo "================================================================================"
echo "Collecting FRED Historical Data (Complete History)"
echo "================================================================================"
python src/ingestion/fred/collect_fred_releases_historical.py

echo ""
echo "================================================================================"
echo "✅ DAILY COLLECTION COMPLETE"
echo "================================================================================"
echo "Completed: $(date)"
echo ""

# Show summary
python -c "
import duckdb
import os

token = os.getenv('MOTHERDUCK_TOKEN')
con = duckdb.connect(f'md:cbi_v15?motherduck_token={token}')

print('Data Summary:')
result = con.execute('''
    SELECT 
        bucket_name,
        COUNT(*) as count,
        MAX(date) as latest_date
    FROM raw.scrapecreators_news_buckets
    GROUP BY bucket_name
    ORDER BY count DESC
''').fetchall()

for row in result:
    print(f'  {row[0]:15} {row[1]:6,} rows  Latest: {row[2]}')

con.close()
"

echo ""
echo "To run daily automatically, add to crontab:"
echo "0 6 * * * /Volumes/Satechi\\ Hub/CBI-V15/scripts/daily_scrapecreators_run.sh >> /tmp/scrapecreators.log 2>&1"


