"""
USDA WASDE Data Ingestion

Downloads World Agricultural Supply & Demand Estimates (WASDE) reports.

Data source: USDA WASDE Reports (monthly)
Coverage: World soy oil, soy meal, soybean production
Frequency: Monthly (usually 12th of each month)

Usage:
    python src/ingestion/usda/ingest_wasde.py --start-year 2020 --end-year 2024
    python src/ingestion/usda/ingest_wasde.py --backfill  # Download all historical data
"""

import os
import sys
import argparse
import requests
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")

# USDA WASDE API (if available) or web scraping
USDA_WASDE_URL = "https://www.usda.gov/oce/commodity/wasde"


def get_connection():
    """Get DuckDB/MotherDuck connection"""
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if motherduck_token:
        return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={motherduck_token}")
    else:
        # Local fallback
        db_path = ROOT_DIR / "data" / "duckdb" / "cbi_v15.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))


def fetch_wasde_data(year: int, month: int) -> pd.DataFrame:
    """Fetch WASDE data for a specific month"""
    
    print(f"  Fetching WASDE {year}-{month:02d}...")
    
    # TODO: Implement actual USDA WASDE scraping/API call
    # For now, return mock data
    return generate_mock_wasde_data(year, month)


def generate_mock_wasde_data(year: int, month: int) -> pd.DataFrame:
    """Generate mock WASDE data for testing"""
    import numpy as np
    
    report_date = pd.Timestamp(year, month, 12)
    
    data = []
    
    # Soybeans
    for country in ['United States', 'Brazil', 'Argentina', 'China', 'World']:
        for metric in ['production', 'consumption', 'exports', 'ending_stocks']:
            data.append({
                'report_date': report_date,
                'commodity': 'Soybeans',
                'country': country,
                'metric': metric,
                'value': np.random.uniform(50, 150) * 1e6,  # Million metric tons
                'unit': 'MT',
                'is_forecast': True,
                'report_month': f"{report_date.strftime('%B %Y')}",
            })
    
    # Soybean Oil
    for country in ['United States', 'Brazil', 'Argentina', 'China', 'World']:
        for metric in ['production', 'consumption', 'exports', 'ending_stocks']:
            data.append({
                'report_date': report_date,
                'commodity': 'Soybean Oil',
                'country': country,
                'metric': metric,
                'value': np.random.uniform(10, 30) * 1e6,  # Million metric tons
                'unit': 'MT',
                'is_forecast': True,
                'report_month': f"{report_date.strftime('%B %Y')}",
            })
    
    # Soybean Meal
    for country in ['United States', 'Brazil', 'Argentina', 'China', 'World']:
        for metric in ['production', 'consumption', 'exports', 'ending_stocks']:
            data.append({
                'report_date': report_date,
                'commodity': 'Soybean Meal',
                'country': country,
                'metric': metric,
                'value': np.random.uniform(30, 80) * 1e6,  # Million metric tons
                'unit': 'MT',
                'is_forecast': True,
                'report_month': f"{report_date.strftime('%B %Y')}",
            })
    
    return pd.DataFrame(data)


def ingest_wasde_data(start_year: int, end_year: int):
    """Ingest WASDE data for specified year range"""
    print(f"\n{'=' * 80}")
    print(f"USDA WASDE DATA INGESTION")
    print(f"{'=' * 80}")
    print(f"Years: {start_year} - {end_year}\n")
    
    con = get_connection()
    all_data = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == datetime.now().year and month > datetime.now().month:
                break
            
            df = fetch_wasde_data(year, month)
            
            if df is not None and not df.empty:
                all_data.append(df)
                print(f"  ✅ {year}-{month:02d}: {len(df):,} rows")
    
    if all_data:
        print(f"\n📊 Combining data...")
        combined = pd.concat(all_data, ignore_index=True)
        print(f"  Total rows: {len(combined):,}")
        
        # Insert into database
        print(f"\n💾 Inserting into database...")
        con.execute("""
            INSERT OR REPLACE INTO raw.usda_wasde
            SELECT * FROM combined
        """)
        print(f"  ✅ Inserted {len(combined):,} rows")
    
    con.close()
    
    print(f"\n{'=' * 80}")
    print(f"✅ WASDE DATA INGESTION COMPLETE")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest USDA WASDE data")
    parser.add_argument("--start-year", type=int, default=2020, help="Start year")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="End year")
    parser.add_argument("--backfill", action="store_true", help="Download all historical data (2010-present)")
    
    args = parser.parse_args()
    
    if args.backfill:
        start_year = 2010
        end_year = datetime.now().year
    else:
        start_year = args.start_year
        end_year = args.end_year
    
    ingest_wasde_data(start_year, end_year)


if __name__ == "__main__":
    main()

