#!/bin/bash
# Daily Weather Refresh - All Sources
# Runs daily to keep weather data current
# Uses FREE APIs only (NOAA, INMET Brazil)

set -e

echo "================================================================================"
echo "DAILY WEATHER REFRESH"
echo "================================================================================"
echo "Started: $(date)"
echo ""

cd "/Volumes/Satechi Hub/CBI-V15"

# Check environment
if [ -z "$NOAA_TOKEN" ]; then
    echo "❌ NOAA_TOKEN not set"
    echo "Get FREE token: https://www.ncdc.noaa.gov/cdo-web/token"
    exit 1
fi

if [ -z "$MOTHERDUCK_TOKEN" ]; then
    echo "❌ MOTHERDUCK_TOKEN not set"
    exit 1
fi

echo "✅ Environment validated"
echo ""

# 1. NOAA Weather (14 stations: US, Brazil, Argentina)
echo "================================================================================"
echo "[1/2] NOAA Weather (14 agricultural regions)"
echo "================================================================================"
python trigger/Weather/Scripts/ingest_weather.py --days 7
echo ""

# 2. INMET Brazil Weather (3 soy regions)
echo "================================================================================"
echo "[2/2] INMET Brazil Weather (3 soy regions)"
echo "================================================================================"
python trigger/Weather/Scripts/collect_brazil_weather.py
echo ""

echo "================================================================================"
echo "✅ DAILY WEATHER REFRESH COMPLETE"
echo "================================================================================"
echo "Completed: $(date)"
echo ""

# Show summary
python -c "
import duckdb
import os

token = os.getenv('MOTHERDUCK_TOKEN')
con = duckdb.connect(f'md:cbi_v15?motherduck_token={token}')

print('Weather Data Summary:')
print('-' * 80)

# NOAA
noaa = con.execute('''
    SELECT 
        country,
        COUNT(DISTINCT station_id) as stations,
        COUNT(*) as days,
        MAX(date) as latest
    FROM raw.weather_noaa
    GROUP BY country
    ORDER BY country
''').fetchall()

for row in noaa:
    print(f'NOAA {row[0]:15} {row[1]} stations  {row[2]:6,} days  Latest: {row[3]}')

# Brazil INMET
try:
    brazil = con.execute('''
        SELECT 
            COUNT(DISTINCT station_id) as stations,
            COUNT(*) as days,
            MAX(date) as latest
        FROM raw.weather_brazil_inmet
    ''').fetchone()
    print(f'INMET Brazil        {brazil[0]} stations  {brazil[1]:6,} days  Latest: {brazil[2]}')
except:
    print('INMET Brazil        Not yet populated')

con.close()
"


