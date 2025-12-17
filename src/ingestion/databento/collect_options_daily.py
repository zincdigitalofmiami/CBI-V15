#!/usr/bin/env python3
"""
Databento Options Daily OHLCV Collection
Collects daily options bars for all futures with listed options.
Target: raw.databento_options_ohlcv_1d

Options symbology: Databento uses instrument IDs for specific contracts.
For options chains, we request data with parent="underlying_symbol" parameter.
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# GLBX.MDP3 dataset
DATASET = "GLBX.MDP3"
DATASET_START = "2010-06-06"

# Symbols with active options markets (prioritize soy complex + energy + metals)
OPTIONS_SYMBOLS = [
    # Soy complex (CRITICAL)
    "ZL",  # Soybean Oil options
    "ZS",  # Soybean options
    "ZM",  # Soybean Meal options
    "ZC",  # Corn options
    "ZW",  # Wheat options
    # Energy (HIGH VOLUME)
    "CL",  # Crude Oil options
    "NG",  # Natural Gas options
    "HO",  # Heating Oil options
    "RB",  # RBOB Gasoline options
    # Metals
    "GC",  # Gold options
    "SI",  # Silver options
    "HG",  # Copper options
    # Rates (volatility/macro)
    "ZN",  # 10-Year Note options
    "ZB",  # 30-Year Bond options
    "ZF",  # 5-Year Note options
    # Equity indices
    "ES",  # S&P 500 options
    "NQ",  # Nasdaq 100 options
    # FX
    "6E",  # Euro options
    "6J",  # Yen options
]


def get_connection():
    """Get MotherDuck connection"""
    if not MOTHERDUCK_TOKEN:
        raise RuntimeError("MOTHERDUCK_TOKEN not set")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")


def get_databento_client():
    """Initialize Databento client"""
    try:
        import databento as db

        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY environment variable not set")
        client = db.Historical(key=api_key)
        logger.info("Initialized Databento client")
        return client
    except ImportError:
        logger.error("databento package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Databento client: {e}")
        return None


def collect_options_for_symbol(
    client, symbol: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Collect options chain data for a specific underlying symbol.

    NOTE: Databento options require specific instrument IDs or parent filtering.
    This is a simplified version - full implementation requires:
    1. Query symbology API to get options instrument IDs for the underlying
    2. Request each options contract individually or use parent filter
    3. Parse options metadata (strike, expiration, type)
    """
    logger.info(f"Collecting options for {symbol} from {start_date} to {end_date}")

    try:
        # Method 1: Try to get options using parent parameter (if supported)
        # This may not work on all Databento subscriptions
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[f"{symbol}.OPT"],  # Options suffix (check Databento docs)
            schema="ohlcv-1d",
            start=start_date,
            end=end_date,
            stype_in="parent",  # Filter by parent underlying
        )

        df = data.to_df().reset_index()

        if df.empty:
            logger.warning(f"No options data for {symbol}")
            return pd.DataFrame()

        # Parse options metadata from symbol
        # Format varies by exchange - this is simplified
        df["symbol"] = symbol
        df["as_of_date"] = pd.to_datetime(df["ts_event"]).dt.date

        return df

    except Exception as e:
        logger.error(f"Failed to collect options for {symbol}: {e}")
        return pd.DataFrame()


def main(days_back: int = 30, dry_run: bool = False):
    """
    Main options ingestion function.

    Args:
        days_back: Days of history to fetch (default 30 for daily incremental)
        dry_run: If True, only log what would be done
    """
    logger.info("=" * 60)
    logger.info("DATABENTO OPTIONS DAILY COLLECTION")
    logger.info("=" * 60)

    client = get_databento_client()
    if not client:
        return

    con = get_connection()
    logger.info(f"Target: MotherDuck")
    logger.info(f"Symbols: {len(OPTIONS_SYMBOLS)}")

    # Determine date range (incremental from last date)
    try:
        result = con.execute(
            """
            SELECT MAX(as_of_date) as last_date
            FROM raw.databento_options_ohlcv_1d
            WHERE symbol = 'ZL'
        """
        ).df()

        if not result.empty and result["last_date"].iloc[0] is not None:
            last_date = pd.to_datetime(result["last_date"].iloc[0]).date()
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"Resuming from {start_date}")
        else:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
            logger.info(f"No existing data. Starting from {start_date}")
    except Exception as e:
        logger.warning(f"Could not check existing data: {e}")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date >= end_date:
        logger.info("Options data is already up to date")
        con.close()
        return

    if dry_run:
        logger.info(f"DRY RUN - would collect {start_date} to {end_date}")
        con.close()
        return

    # Collect options for each symbol
    all_options = []
    for symbol in OPTIONS_SYMBOLS:
        options_df = collect_options_for_symbol(client, symbol, start_date, end_date)
        if not options_df.empty:
            all_options.append(options_df)

    if not all_options:
        logger.warning("No options data collected")
        con.close()
        return

    # Combine and load
    combined = pd.concat(all_options, ignore_index=True)
    logger.info(f"Total options contracts: {len(combined):,}")

    # Load to MotherDuck (simplified - actual schema mapping needed)
    con.register("options_staging", combined)
    con.execute(
        """
        DELETE FROM raw.databento_options_ohlcv_1d
        WHERE (contract_symbol, as_of_date) IN (
            SELECT contract_symbol, as_of_date FROM options_staging
        )
    """
    )
    con.execute(
        """
        INSERT INTO raw.databento_options_ohlcv_1d
            (symbol, contract_symbol, as_of_date, open, high, low, close, volume)
        SELECT symbol, symbol as contract_symbol, as_of_date, 
               open, high, low, close, volume
        FROM options_staging
    """
    )

    logger.info(
        f"Successfully loaded {len(combined):,} rows to raw.databento_options_ohlcv_1d"
    )
    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Days to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Dry run only")
    args = parser.parse_args()

    main(days_back=args.days, dry_run=args.dry_run)


