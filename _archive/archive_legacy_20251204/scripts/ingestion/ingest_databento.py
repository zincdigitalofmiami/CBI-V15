#!/usr/bin/env python3
"""
Databento Ingestion Script (AnoFox Pipeline)
Fetches OHLCV data, validates schema, and writes to partitioned Parquet.
"""

import os
import sys
import duckdb
import databento as db
import pandas as pd
from pathlib import Path
from datetime import datetime, date

# Configuration
API_KEY = os.getenv("DATABENTO_API_KEY")
RAW_DIR = Path("Data/raw/databento")
PARQUET_DIR = Path("Data/parquet/market")
DB_PATH = Path("Data/db/cbi-v15.duckdb")

SYMBOLS = ["ZL.FUT", "ZS.FUT", "ZM.FUT", "HE.FUT", "CL.FUT"] 

def fetch_data(symbol, start_date, end_date):
    """Fetch DBN from Databento."""
    client = db.Historical(key=API_KEY)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        schema="ohlcv-1d",
        start=start_date,
        end=end_date
    )
    return data.to_df()

def process_and_store(df, symbol):
    """Standardize schema and save to Parquet."""
    df.reset_index(inplace=True)
    df['symbol'] = symbol
    
    # Schema enforcement
    required_cols = ['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    df = df.rename(columns={'ts_event': 'date'})
    
    # Type casting
    df['date'] = pd.to_datetime(df['date'])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(int)

    # Partitioned Write via DuckDB
    con = duckdb.connect()
    con.execute("CREATE TABLE temp_ingest AS SELECT * FROM df")
    
    out_path = PARQUET_DIR / symbol.replace(".", "_")
    os.makedirs(out_path, exist_ok=True)
    
    con.execute(f"""
        COPY temp_ingest TO '{out_path}' 
        (FORMAT PARQUET, PARTITION_BY (date), OVERWRITE_OR_IGNORE 1)
    """)
    print(f"✅ Ingested {len(df)} rows for {symbol}")

if __name__ == "__main__":
    if not API_KEY:
        print("❌ Error: DATABENTO_API_KEY not set.")
        sys.exit(1)
        
    for sym in SYMBOLS:
        print(f"Fetching {sym}...")
        try:
            df = fetch_data(sym, "2020-01-01", date.today().isoformat())
            if not df.empty:
                process_and_store(df, sym)
        except Exception as e:
            print(f"⚠️ Failed to fetch {sym}: {e}")

