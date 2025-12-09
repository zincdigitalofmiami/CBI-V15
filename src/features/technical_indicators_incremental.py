#!/usr/bin/env python3
"""
Technical Indicators Calculation - Incremental Updates
Uses pandas-ta for daily updates

NOTE: This is a LEGACY script. Technical indicators should be computed
via SQL macros in database/macros/technical_indicators_all_symbols.sql.
Use src/engines/anofox/build_all_features.py instead.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pandas_ta as ta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Database config
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
LOCAL_DB_PATH = project_root / "Data" / "duckdb" / "local.duckdb"


def get_connection(use_motherduck: bool = True):
    """Get DuckDB connection (MotherDuck or local)"""
    if use_motherduck and MOTHERDUCK_TOKEN:
        return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
    else:
        return duckdb.connect(str(LOCAL_DB_PATH))


def calculate_indicators_incremental(
    symbol: str = "ZL", days_back: int = 1, use_motherduck: bool = True
):
    """Calculate indicators for new data only using pandas-ta

    NOTE: Prefer using SQL macros via build_all_features.py instead.
    This script is kept for compatibility but features should be in SQL.
    """

    con = get_connection(use_motherduck)

    # Get new data only
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    query = f"""
    SELECT 
      date,
      symbol,
      close
    FROM staging.market_daily
    WHERE symbol = '{symbol}' 
      AND date >= '{start_date}'
    ORDER BY date
    """

    print(f"Fetching data for {symbol} from {start_date}...")
    df = con.execute(query).df()

    if df.empty:
        print(f"⚠️  No new data for {symbol}")
        con.close()
        return

    # Set date as index
    df.set_index("date", inplace=True)

    # Calculate indicators using pandas-ta (close-only)
    print(f"Calculating technical indicators...")

    # RSI
    df.ta.rsi(close="close", length=14, append=True)

    # SMA (Moving Averages)
    df.ta.sma(close="close", length=10, append=True)
    df.ta.sma(close="close", length=20, append=True)
    df.ta.sma(close="close", length=50, append=True)

    # EMA
    df.ta.ema(close="close", length=21, append=True)

    # Reset index
    df.reset_index(inplace=True)

    # Insert to DuckDB
    print(f"Inserting {len(df)} rows to features.technical_indicators_incremental...")

    # Create table if not exists
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS features.technical_indicators_incremental (
            date DATE,
            symbol VARCHAR,
            close DOUBLE,
            RSI_14 DOUBLE,
            SMA_10 DOUBLE,
            SMA_20 DOUBLE,
            SMA_50 DOUBLE,
            EMA_21 DOUBLE
        )
    """
    )

    # Insert data
    con.execute(
        """
        INSERT INTO features.technical_indicators_incremental 
        SELECT * FROM df
    """
    )

    con.close()
    print(f"✅ Indicators calculated and uploaded for {symbol}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ZL"
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    use_md = "--local" not in sys.argv
    calculate_indicators_incremental(symbol, days_back, use_md)
