#!/usr/bin/env python3
"""
US Corn Belt Weather Collection - NOAA CDO API

Direct API pull. NO Trigger.dev.

Regions covered:
- Iowa (IA) - Core corn/soy production
- Illinois (IL) - #1 soybean state
- Nebraska (NE) - Major corn producer
- Minnesota (MN) - Northern corn belt
- Indiana (IN) - Eastern corn belt

API: https://www.ncdc.noaa.gov/cdo-web/api/v2/
Token: Free from https://www.ncdc.noaa.gov/cdo-web/token

Usage:
    python collect_us_cornbelt.py                    # Yesterday's data
    python collect_us_cornbelt.py --days 7           # Last 7 days
    python collect_us_cornbelt.py --backfill 1000    # Backfill 1000 days
"""

import os
import sys
import argparse
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[3] / ".env")

import duckdb

# Configuration
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
NOAA_TOKEN = os.getenv("NOAA_API_TOKEN")
NOAA_API_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# US Corn Belt Stations
# Format: (station_id, region_code, state_name)
STATIONS = [
    # Iowa - Core corn/soy
    ("GHCND:USC00130200", "US_IA", "Iowa"),      # Ames
    ("GHCND:USC00132724", "US_IA", "Iowa"),      # Des Moines
    ("GHCND:USC00137147", "US_IA", "Iowa"),      # Sioux City
    ("GHCND:USC00131533", "US_IA", "Iowa"),      # Cedar Rapids
    ("GHCND:USC00134735", "US_IA", "Iowa"),      # Iowa City
    
    # Illinois - #1 soybean state
    ("GHCND:USC00111577", "US_IL", "Illinois"),  # Champaign
    ("GHCND:USC00111290", "US_IL", "Illinois"),  # Cairo
    ("GHCND:USC00118740", "US_IL", "Illinois"),  # Springfield
    ("GHCND:USC00114442", "US_IL", "Illinois"),  # Galesburg
    ("GHCND:USC00116711", "US_IL", "Illinois"),  # Peoria
    
    # Nebraska - Major corn
    ("GHCND:USC00255362", "US_NE", "Nebraska"),  # Lincoln
    ("GHCND:USC00256135", "US_NE", "Nebraska"),  # Omaha
    ("GHCND:USC00254795", "US_NE", "Nebraska"),  # Kearney
    ("GHCND:USC00254110", "US_NE", "Nebraska"),  # Grand Island
    ("GHCND:USC00257070", "US_NE", "Nebraska"),  # Scottsbluff
    
    # Minnesota - Northern corn belt
    ("GHCND:USC00215435", "US_MN", "Minnesota"), # Minneapolis
    ("GHCND:USC00218450", "US_MN", "Minnesota"), # St. Paul
    ("GHCND:USC00214884", "US_MN", "Minnesota"), # Mankato
    ("GHCND:USC00217107", "US_MN", "Minnesota"), # Rochester
    ("GHCND:USC00212916", "US_MN", "Minnesota"), # Duluth
    
    # Indiana - Eastern corn belt
    ("GHCND:USC00124259", "US_IN", "Indiana"),   # Indianapolis
    ("GHCND:USC00124837", "US_IN", "Indiana"),   # Lafayette
    ("GHCND:USC00121747", "US_IN", "Indiana"),   # Columbus
    ("GHCND:USC00123418", "US_IN", "Indiana"),   # Fort Wayne
    ("GHCND:USC00128784", "US_IN", "Indiana"),   # Terre Haute
]

COUNTRY = "United States"


def get_connection():
    """Get MotherDuck connection"""
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}")


def fetch_station_data(station_id: str, start_date: str, end_date: str) -> list:
    """Fetch GHCND data for a station from NOAA CDO API"""
    
    if not NOAA_TOKEN:
        raise RuntimeError("NOAA_API_TOKEN not set in .env")
    
    headers = {"token": NOAA_TOKEN}
    all_results = []
    offset = 1
    
    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": "TMAX,TMIN,TAVG,PRCP,SNOW",
            "units": "metric",
            "limit": 1000,
            "offset": offset
        }
        
        try:
            resp = requests.get(f"{NOAA_API_BASE}/data", headers=headers, params=params, timeout=60)
            
            if resp.status_code == 429:
                print(f"      Rate limited, waiting 2s...")
                time.sleep(2)
                continue
                
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("results", [])
            if not results:
                break
                
            all_results.extend(results)
            
            # Check if more pages
            metadata = data.get("metadata", {}).get("resultset", {})
            total_count = metadata.get("count", 0)
            
            if offset + len(results) >= total_count:
                break
            
            offset += 1000
            time.sleep(0.3)  # Rate limit: 5 requests/second
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                # Station may not have data for this period
                break
            raise
    
    return all_results


def process_station_data(results: list, station_id: str, region: str) -> list:
    """Convert NOAA API results to our schema"""
    
    # Group by date
    by_date = {}
    for r in results:
        date = r["date"][:10]
        if date not in by_date:
            by_date[date] = {}
        by_date[date][r["datatype"]] = r["value"]
    
    # Convert to records
    records = []
    for date, values in by_date.items():
        records.append({
            "station_id": station_id,
            "date": date,
            "tavg_c": values.get("TAVG"),
            "tmin_c": values.get("TMIN"),
            "tmax_c": values.get("TMAX"),
            "prcp_mm": values.get("PRCP"),
            "snow_mm": values.get("SNOW"),
            "region": region,
            "country": COUNTRY,
            "source": "noaa_cdo"
        })
    
    return records


def collect_us_cornbelt(start_date: str, end_date: str) -> int:
    """Collect weather data from US Corn Belt stations"""
    
    print(f"\n🇺🇸 US CORN BELT WEATHER COLLECTION")
    print(f"   Stations: {len(STATIONS)}")
    print(f"   Date range: {start_date} to {end_date}\n")
    
    con = get_connection()
    total_inserted = 0
    
    for station_id, region, state in STATIONS:
        print(f"   📍 {station_id} ({state})...", end=" ", flush=True)
        
        try:
            results = fetch_station_data(station_id, start_date, end_date)
            
            if not results:
                print("⚠️ No data")
                continue
            
            records = process_station_data(results, station_id, region)
            
            if records:
                # Insert into MotherDuck
                for r in records:
                    con.execute("""
                        INSERT OR REPLACE INTO raw.weather_noaa 
                        (station_id, date, tavg_c, tmin_c, tmax_c, prcp_mm, snow_mm, region, country, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [r["station_id"], r["date"], r["tavg_c"], r["tmin_c"], r["tmax_c"],
                          r["prcp_mm"], r["snow_mm"], r["region"], r["country"], r["source"]])
                
                print(f"✅ {len(records)} rows")
                total_inserted += len(records)
            
            time.sleep(0.5)  # Be nice to the API
            
        except Exception as e:
            print(f"❌ {e}")
    
    con.close()
    print(f"\n   🇺🇸 US Total: {total_inserted} rows inserted\n")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Collect US Corn Belt weather data")
    parser.add_argument("--days", type=int, default=1, help="Number of days to fetch (default: 1)")
    parser.add_argument("--backfill", type=int, help="Backfill N days of history")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    
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
    
    collect_us_cornbelt(start_date, end_date)


if __name__ == "__main__":
    main()
