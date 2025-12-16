#!/usr/bin/env python3
"""
Argentina Pampas Weather Collection - NOAA CDO API

Direct API pull. NO Trigger.dev.

Regions covered:
- Buenos Aires (BA) - Largest agricultural province
- Córdoba (CO) - Second largest soy/corn
- Santa Fe (SF) - Major soy processor region
- Entre Ríos (ER) - Eastern agricultural zone
- La Pampa (LP) - Western wheat/cattle

API: https://www.ncdc.noaa.gov/cdo-web/api/v2/
Token: Free from https://www.ncdc.noaa.gov/cdo-web/token

Usage:
    python collect_argentina_pampas.py                    # Yesterday's data
    python collect_argentina_pampas.py --days 7           # Last 7 days
    python collect_argentina_pampas.py --backfill 1000    # Backfill 1000 days
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

# Argentina Pampas Stations - VERIFIED ACTIVE (data through Aug 2025)
# Format: (station_id, region_code, province_name)
STATIONS = [
    # Buenos Aires Province - Largest ag province
    ("GHCND:AR000875850", "AR_BA", "Buenos Aires"),     # Buenos Aires Observ (1908-present)
    ("GHCND:AR000877500", "AR_BA", "Buenos Aires"),     # Bahía Blanca Aero
    ("GHCND:AR000087692", "AR_BA", "Buenos Aires"),     # Mar del Plata Aero
    ("GHCND:AR000875440", "AR_BA", "Buenos Aires"),     # Pehuajó Aero (corn belt)
    
    # Córdoba - Second largest soy/corn
    ("GHCND:AR000087344", "AR_CO", "Córdoba"),          # Córdoba Aero
    ("GHCND:AR000087534", "AR_CO", "Córdoba"),          # Laboulaye Aero (ag region)
    
    # Santa Fe / Entre Ríos
    ("GHCND:AR000087257", "AR_SF", "Santa Fe"),         # Ceres Aero
    ("GHCND:AR000087374", "AR_ER", "Entre Ríos"),       # Paraná Aero
    ("GHCND:AR000000011", "AR_ER", "Entre Ríos"),       # Monte Caseros Aero
    ("GHCND:ARM00087166", "AR_CR", "Corrientes"),       # Corrientes
    
    # La Pampa - Western wheat/cattle
    ("GHCND:AR000087623", "AR_LP", "La Pampa"),         # Santa Rosa Aero
    
    # Northern regions (Chaco, Formosa - expanding ag)
    ("GHCND:AR000087155", "AR_CH", "Chaco"),            # Resistencia Aero
    ("GHCND:ARM00087162", "AR_FO", "Formosa"),          # Formosa
    ("GHCND:ARM00087148", "AR_CH", "Chaco"),            # Presidencia Roque Sáenz Peña
    
    # Mendoza (wine/fruit, not soy but important)
    ("GHCND:AR000087418", "AR_MZ", "Mendoza"),          # Mendoza Aero
    
    # Santiago del Estero (expanding soy frontier)
    ("GHCND:AR000087129", "AR_SE", "Santiago del Estero"), # Santiago del Estero
]

COUNTRY = "Argentina"


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
            
            metadata = data.get("metadata", {}).get("resultset", {})
            total_count = metadata.get("count", 0)
            
            if offset + len(results) >= total_count:
                break
            
            offset += 1000
            time.sleep(0.3)
            
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


def collect_argentina_pampas(start_date: str, end_date: str) -> int:
    """Collect weather data from Argentina Pampas stations"""
    
    print(f"\n🇦🇷 ARGENTINA PAMPAS WEATHER COLLECTION")
    print(f"   Stations: {len(STATIONS)}")
    print(f"   Date range: {start_date} to {end_date}\n")
    
    con = get_connection()
    total_inserted = 0
    
    for station_id, region, province in STATIONS:
        print(f"   📍 {station_id} ({province})...", end=" ", flush=True)
        
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
    print(f"\n   🇦🇷 Argentina Total: {total_inserted} rows inserted\n")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Collect Argentina Pampas weather data")
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
    
    collect_argentina_pampas(start_date, end_date)


if __name__ == "__main__":
    main()
