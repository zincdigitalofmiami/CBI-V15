#!/usr/bin/env python3
"""
Brazil Soy Belt Weather Collection - NOAA CDO API

Direct API pull (no external orchestrator coupling).

Regions covered:
- Mato Grosso (MT) - Largest soy state (~30% of Brazil production)
- Goiás (GO) - Second largest soy state
- Paraná (PR) - Major soy/corn state
- Rio Grande do Sul (RS) - Southern soy production
- Mato Grosso do Sul (MS) - Growing soy region
- Bahia (BA) - MATOPIBA region (emerging)

API: https://www.ncdc.noaa.gov/cdo-web/api/v2/
Token: Free from https://www.ncdc.noaa.gov/cdo-web/token

Usage:
    python collect_brazil_soy_belt.py                    # Yesterday's data
    python collect_brazil_soy_belt.py --days 7           # Last 7 days
    python collect_brazil_soy_belt.py --backfill 1000    # Backfill 1000 days
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

# Brazil Soy Belt Stations - VERIFIED ACTIVE (data through Aug 2025)
# Format: (station_id, region_code, state_name)
STATIONS = [
    # Mato Grosso - Largest soy state
    ("GHCND:BR000956000", "BR_MT", "Mato Grosso"),      # Alta Floresta Aero
    ("GHCND:BR000352000", "BR_MT", "Mato Grosso"),      # Altamira (nearby)
    
    # Paraná - Major soy/corn
    ("GHCND:BR002351003", "BR_PR", "Paraná"),           # Londrina Aeroporto
    ("GHCND:BR002549075", "BR_PR", "Paraná"),           # Curitiba
    ("GHCND:BR002548005", "BR_PR", "Paraná"),           # Paranaguá
    
    # Rio Grande do Sul - Southern production
    ("GHCND:BR002752004", "BR_RS", "Rio Grande do Sul"), # Chapecó
    
    # São Paulo region (proxy for GO/MS)
    ("GHCND:BR00E3-0520", "BR_SP", "São Paulo"),        # São Paulo Aeroporto
    ("GHCND:BR00D6-0010", "BR_SP", "São Paulo"),        # Bauru
    ("GHCND:BR00D8-0390", "BR_SP", "São Paulo"),        # Presidente Prudente
    
    # Minas Gerais (adjacent to GO)
    ("GHCND:BR001943012", "BR_MG", "Minas Gerais"),     # Lagoa Santa Airport
    
    # Mato Grosso do Sul
    ("GHCND:BR001957002", "BR_MS", "Mato Grosso do Sul"), # Corumbá
    
    # Northeast (Bahia proxy)
    ("GHCND:BR000082400", "BR_NE", "Northeast"),        # Fernando de Noronha
    ("GHCND:BR000254000", "BR_PA", "Pará"),             # Santarém (Amazon soy)
]

COUNTRY = "Brazil"


def get_connection():
    """Get MotherDuck connection"""
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
        return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={token}")
    raise RuntimeError(
        "MotherDuck token required (set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN)"
    )


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
                break
            raise
    
    return all_results


def process_station_data(results: list, station_id: str, region: str) -> list:
    """Convert NOAA API results to our schema"""
    
    by_date = {}
    for r in results:
        date = r["date"][:10]
        if date not in by_date:
            by_date[date] = {}
        by_date[date][r["datatype"]] = r["value"]
    
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


def collect_brazil_soy_belt(start_date: str, end_date: str) -> int:
    """Collect weather data from Brazil Soy Belt stations"""
    
    print(f"\n🇧🇷 BRAZIL SOY BELT WEATHER COLLECTION")
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
                for r in records:
                    con.execute("""
                        INSERT OR REPLACE INTO raw.weather_noaa 
                        (station_id, date, tavg_c, tmin_c, tmax_c, prcp_mm, snow_mm, region, country, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [r["station_id"], r["date"], r["tavg_c"], r["tmin_c"], r["tmax_c"],
                          r["prcp_mm"], r["snow_mm"], r["region"], r["country"], r["source"]])
                
                print(f"✅ {len(records)} rows")
                total_inserted += len(records)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ {e}")
    
    con.close()
    print(f"\n   🇧🇷 Brazil Total: {total_inserted} rows inserted\n")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Collect Brazil Soy Belt weather data")
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
    
    collect_brazil_soy_belt(start_date, end_date)


if __name__ == "__main__":
    main()
