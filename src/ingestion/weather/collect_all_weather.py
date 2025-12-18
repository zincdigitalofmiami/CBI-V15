#!/usr/bin/env python3
"""
NOAA Weather Collection Orchestrator

Calls all country-specific weather collection scripts.
Direct API pulls (no external orchestrator coupling).

Coverage:
- 🇺🇸 US Corn Belt (25 stations): Iowa, Illinois, Nebraska, Minnesota, Indiana
- 🇧🇷 Brazil Soy Belt (23 stations): MT, GO, PR, RS, MS, BA
- 🇦🇷 Argentina Pampas (18 stations): BA, CO, SF, ER, LP

Total: 66 agricultural weather stations

Usage:
    python collect_all_weather.py                    # Yesterday's data
    python collect_all_weather.py --days 7           # Last 7 days
    python collect_all_weather.py --backfill 1000    # Backfill 1000 days
    python collect_all_weather.py --country US       # Just US
    python collect_all_weather.py --country Brazil   # Just Brazil
    python collect_all_weather.py --country Argentina # Just Argentina
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
WEATHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(WEATHER_DIR / "US"))
sys.path.insert(0, str(WEATHER_DIR / "Brazil"))
sys.path.insert(0, str(WEATHER_DIR / "Argentina"))

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[3] / ".env")

import duckdb

# Import country collectors
from collect_us_cornbelt import collect_us_cornbelt
from collect_brazil_soy_belt import collect_brazil_soy_belt
from collect_argentina_pampas import collect_argentina_pampas


def get_connection():
    """Get MotherDuck connection"""
    db = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    candidates = [
        os.getenv("MOTHERDUCK_TOKEN"),
        os.getenv("motherduck_storage_MOTHERDUCK_TOKEN"),
    ]
    for raw in candidates:
        if not raw:
            continue
        token = raw.strip().strip('"').strip("'")
        if token.count(".") != 2:
            continue
        return duckdb.connect(f"md:{db}?motherduck_token={token}")
    raise RuntimeError(
        "MotherDuck token required (set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN)"
    )


def collect_all_weather(start_date: str, end_date: str, country: str = None):
    """
    Collect weather data from all agricultural regions.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        country: Optional filter (US, Brazil, Argentina)
    """
    
    print(f"\n{'='*70}")
    print(f"🌍 NOAA AGRICULTURAL WEATHER COLLECTION")
    print(f"{'='*70}")
    print(f"Date range: {start_date} to {end_date}")
    if country:
        print(f"Country filter: {country}")
    print(f"{'='*70}\n")
    
    total_rows = 0
    
    # US Corn Belt
    if not country or country.upper() == "US":
        us_rows = collect_us_cornbelt(start_date, end_date)
        total_rows += us_rows
    
    # Brazil Soy Belt
    if not country or country.upper() == "BRAZIL":
        brazil_rows = collect_brazil_soy_belt(start_date, end_date)
        total_rows += brazil_rows
    
    # Argentina Pampas
    if not country or country.upper() == "ARGENTINA":
        argentina_rows = collect_argentina_pampas(start_date, end_date)
        total_rows += argentina_rows
    
    # Final summary
    con = get_connection()
    result = con.execute("SELECT COUNT(*) FROM raw.weather_noaa").fetchone()
    
    # Get country breakdown
    breakdown = con.execute("""
        SELECT country, COUNT(*) as cnt 
        FROM raw.weather_noaa 
        GROUP BY country 
        ORDER BY cnt DESC
    """).fetchall()
    
    # Get date range
    date_range = con.execute("""
        SELECT MIN(date) as min_date, MAX(date) as max_date 
        FROM raw.weather_noaa
    """).fetchone()
    
    con.close()
    
    print(f"\n{'='*70}")
    print(f"✅ WEATHER COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"   New rows inserted: {total_rows:,}")
    print(f"   Total rows in table: {result[0]:,}")
    print(f"\n   By Country:")
    for country, cnt in breakdown:
        print(f"      {country}: {cnt:,} rows")
    print(f"\n   Date coverage: {date_range[0]} to {date_range[1]}")
    print(f"{'='*70}\n")
    
    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Collect NOAA agricultural weather data")
    parser.add_argument("--days", type=int, default=1, help="Number of days to fetch (default: 1)")
    parser.add_argument("--backfill", type=int, help="Backfill N days of history")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--country", type=str, choices=["US", "Brazil", "Argentina"], 
                        help="Collect only specific country")
    
    args = parser.parse_args()
    
    if args.backfill:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.backfill)).strftime("%Y-%m-%d")
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    collect_all_weather(start_date, end_date, args.country)


if __name__ == "__main__":
    main()
