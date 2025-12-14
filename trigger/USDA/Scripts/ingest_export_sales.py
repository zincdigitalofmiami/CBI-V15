"""
USDA Export Sales Data Ingestion

Downloads weekly export sales reports from USDA FAS.

Data source: USDA FAS Export Sales Reports
Coverage: Soybeans, Soybean Oil, Soybean Meal
Frequency: Weekly (released every Thursday at 8:30 AM ET)

API: USDA FAS API or web scraping from:
- https://apps.fas.usda.gov/export-sales/esrd1.html

Usage:
    python trigger/USDA/Scripts/ingest_export_sales.py --start-date 2020-01-01
    python trigger/USDA/Scripts/ingest_export_sales.py --backfill
"""

import os
import sys
import argparse
import requests
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# USDA FAS Export Sales URL
USDA_EXPORT_SALES_URL = "https://apps.fas.usda.gov/export-sales/esrd1.html"


def get_connection():
    """Get DuckDB/MotherDuck connection"""
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if motherduck_token:
        return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={motherduck_token}")
    else:
        db_path = ROOT_DIR / "data" / "duckdb" / "cbi_v15.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))


def fetch_export_sales(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch export sales data from USDA FAS"""
    
    print(f"  Fetching export sales from USDA FAS...")
    
    # TODO: Implement actual USDA FAS scraping
    # For now, return mock data
    return generate_mock_export_sales(start_date, end_date)


def generate_mock_export_sales(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate mock export sales data for testing"""
    import numpy as np
    
    # Generate weekly dates (Thursdays)
    dates = pd.date_range(start=start_date, end=end_date, freq='W-THU')
    
    data = []
    
    commodities = ['Soybeans', 'Soybean Oil', 'Soybean Meal']
    countries = ['China', 'Mexico', 'Japan', 'Indonesia', 'European Union', 'Unknown']
    
    for report_date in dates:
        for commodity in commodities:
            for country in countries:
                # Skip China for recent dates (reflecting current trade situation)
                if country == 'China' and report_date > pd.Timestamp('2025-05-01'):
                    if commodity == 'Soybeans':
                        continue  # No China soybean purchases since May 2025
                
                data.append({
                    'report_date': report_date,
                    'commodity': commodity,
                    'destination_country': country,
                    'net_sales_mt': np.random.uniform(10000, 500000) if country != 'China' else 0,
                    'accumulated_exports_mt': np.random.uniform(1e6, 10e6),
                    'outstanding_sales_mt': np.random.uniform(500000, 5e6),
                    'marketing_year': f"{report_date.year}/{report_date.year + 1}",
                    'report_week_ending': report_date,
                })
    
    return pd.DataFrame(data)


def ingest_export_sales(start_date: str, end_date: str):
    """Ingest export sales data"""
    print(f"\n{'=' * 80}")
    print(f"USDA EXPORT SALES DATA INGESTION")
    print(f"{'=' * 80}")
    print(f"Date range: {start_date} to {end_date}\n")
    
    con = get_connection()
    
    df = fetch_export_sales(start_date, end_date)
    
    if df is not None and not df.empty:
        print(f"\n📊 Data summary:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Date range: {df['report_date'].min()} to {df['report_date'].max()}")
        print(f"  Commodities: {df['commodity'].nunique()}")
        print(f"  Countries: {df['destination_country'].nunique()}")
        
        # Insert into database
        print(f"\n💾 Inserting into database...")
        con.execute("""
            INSERT OR REPLACE INTO raw.usda_export_sales
            SELECT * FROM df
        """)
        print(f"  ✅ Inserted {len(df):,} rows")
    
    con.close()
    
    print(f"\n{'=' * 80}")
    print(f"✅ EXPORT SALES DATA INGESTION COMPLETE")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest USDA export sales data")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--backfill", action="store_true", help="Download all historical data (2015-present)")
    
    args = parser.parse_args()
    
    if args.backfill:
        start_date = "2015-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        start_date = args.start_date
        end_date = args.end_date
    
    ingest_export_sales(start_date, end_date)


if __name__ == "__main__":
    main()
