#!/usr/bin/env python3
"""
Collect daily OHLCV futures data from Databento API
Stores to: raw.databento_futures_ohlcv_1d

NOTE: This stores to DuckDB/MotherDuck, NOT BigQuery.
"""
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.keychain_manager import get_api_key

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
LOCAL_DB_PATH = project_root / "data" / "duckdb" / "cbi_v15.duckdb"

# Symbols to collect (38 total as per SYSTEM_STATUS)
SYMBOLS = [
    # Agricultural (11)
    "ZL",
    "ZS",
    "ZM",
    "ZC",
    "ZW",
    "ZO",
    "ZR",
    "HE",
    "LE",
    "GF",
    "FCPO",
    # Energy (4)
    "CL",
    "HO",
    "RB",
    "NG",
    # Metals (5)
    "HG",
    "GC",
    "SI",
    "PL",
    "PA",
    # Treasuries (3)
    "ZF",
    "ZN",
    "ZB",
    # FX Futures (10)
    "6E",
    "6J",
    "6B",
    "6C",
    "6A",
    "6N",
    "6M",
    "6L",
    "6S",
    "DX",
]


def get_connection(use_motherduck: bool = True):
    """Get DuckDB connection (MotherDuck or local)"""
    if use_motherduck and MOTHERDUCK_TOKEN:
        return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
    else:
        LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(LOCAL_DB_PATH))


def get_databento_client():
    """Initialize Databento client"""
    try:
        import databento as db

        api_key = get_api_key("DATABENTO_API_KEY")
        if not api_key:
            api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not found in Keychain or environment")
        client = db.Historical(key=api_key)
        return client
    except ImportError:
        logger.error(
            "databento package not installed. Install with: pip install databento"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Databento client: {e}")
        return None


def collect_daily_data(
    symbol: str, start_date: str, end_date: str = None
) -> pd.DataFrame:
    """
    Collect daily OHLCV data for a symbol

    Args:
        symbol: Futures symbol (e.g., 'ZL')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with columns: as_of_date, symbol, open, high, low, close, volume, open_interest
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    client = get_databento_client()
    if not client:
        return pd.DataFrame()

    try:
        logger.info(f"Collecting {symbol} data from {start_date} to {end_date}")

        # Databento API call
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            schema="ohlcv-1d",
            start=start_date,
            end=end_date,
        )

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Standardize column names to match raw.databento_futures_ohlcv_1d schema
        df = df.rename(
            columns={
                "ts_event": "as_of_date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "open_interest": "open_interest",
            }
        )

        # Add symbol column
        df["symbol"] = symbol

        # Ensure as_of_date is DATE type
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date

        # Select columns in order
        df = df[
            [
                "as_of_date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        ]

        logger.info(f"Collected {len(df)} rows for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error collecting {symbol}: {e}")
        return pd.DataFrame()


def main(use_motherduck: bool = True):
    """Main ingestion function"""
    logger.info("🚀 Starting Databento daily data collection")

    con = get_connection(use_motherduck)
    target = "MotherDuck" if use_motherduck else "Local DuckDB"
    logger.info(f"Target: {target}")

    # Get last date from DuckDB
    try:
        result = con.execute(
            """
            SELECT MAX(as_of_date) as last_date
            FROM raw.databento_futures_ohlcv_1d
            WHERE symbol = 'ZL'
        """
        ).df()
        if not result.empty and result["last_date"].iloc[0] is not None:
            last_date = pd.to_datetime(result["last_date"].iloc[0]).date()
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"Resuming from {start_date}")
        else:
            start_date = "2010-01-01"
            logger.info(f"No existing data, starting from {start_date}")
    except Exception as e:
        logger.warning(f"Could not determine last date: {e}. Starting from 2010-01-01")
        start_date = "2010-01-01"

    end_date = datetime.now().strftime("%Y-%m-%d")

    # Collect data for all symbols
    all_data = []
    for symbol in SYMBOLS:
        df = collect_daily_data(symbol, start_date, end_date)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        logger.warning("No data collected")
        con.close()
        return

    # Combine all symbols
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total rows collected: {len(combined_df)}")

    # Load to DuckDB
    try:
        # Idempotent load: replace on primary key (symbol, as_of_date)
        con.execute(
            """
            CREATE TEMP TABLE staging_load AS SELECT * FROM combined_df;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_stage_pk ON staging_load(symbol, as_of_date);
            INSERT OR REPLACE INTO raw.databento_futures_ohlcv_1d
            SELECT symbol, as_of_date, open, high, low, close, volume, open_interest
            FROM staging_load;
        """
        )
        logger.info(
            f"✅ Successfully loaded {len(combined_df)} rows to raw.databento_futures_ohlcv_1d"
        )
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    use_md = "--local" not in sys.argv
    main(use_motherduck=use_md)
