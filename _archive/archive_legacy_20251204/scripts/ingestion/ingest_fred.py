#!/usr/bin/env python3
"""
FRED Ingestion Script (AnoFox Pipeline)
Fetches economic series, pivots to wide format, and saves to Parquet.
"""

import os
import sys
import pandas as pd
import pandas_datareader.data as web
from pathlib import Path
from datetime import datetime

# Configuration
PARQUET_DIR = Path("Data/parquet/macro")
START_DATE = "2000-01-01"

SERIES_MAP = {
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "DTWEXBGS": "dollar_index_broad",
    "BAMLH0A0HYM2": "high_yield_spread",
    "T10Y2Y": "yield_curve_10y2y",
    "DCOILWTICO": "wti_crude",
    "DCOILBRENTEU": "brent_crude"
}

def ingest_fred():
    """Fetch all series and combine."""
    print("Fetching FRED data...")
    try:
        df = web.DataReader(list(SERIES_MAP.keys()), "fred", START_DATE, datetime.today())
        df = df.rename(columns=SERIES_MAP)
        df.index.name = "date"
        df.reset_index(inplace=True)
        
        # Save
        os.makedirs(PARQUET_DIR, exist_ok=True)
        out_file = PARQUET_DIR / "fred_macro_daily.parquet"
        df.to_parquet(out_file)
        print(f"✅ Saved {len(df)} rows to {out_file}")
        
    except Exception as e:
        print(f"❌ FRED Ingestion failed: {e}")

if __name__ == "__main__":
    ingest_fred()

