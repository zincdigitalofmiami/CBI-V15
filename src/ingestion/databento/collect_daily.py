#!/usr/bin/env python3
"""
Collect daily OHLCV futures data from Databento API
Stores to: raw.databento_futures_ohlcv_1d

BEST PRACTICES (per Databento docs):
1. Batch multiple symbols in ONE request (up to 2,000 symbols)
2. Estimate cost before large requests using metadata.get_cost
3. Use exponential backoff for rate limit handling
4. Use stype_in="continuous" for continuous contract symbology

NOTE: This stores to MotherDuck cloud, NOT BigQuery.
"""
import logging
import os
import sys
import time
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
# NOTE: No local DB path - all data goes to MotherDuck cloud

# GLBX.MDP3 dataset availability
DATASET = "GLBX.MDP3"
DATASET_START = "2010-06-06"  # Earliest available date for GLBX.MDP3

# Master Symbol List - CME Group (GLBX.MDP3) ONLY
# VALIDATED: All symbols below resolve in GLBX.MDP3 as of Dec 2025
# Updates:
# - ZE removed; use CU (Chicago Ethanol)
# - DA removed; use DC (Class III) + DY (Dry Whey)
# - MSI corrected to SIL (Micro Silver)
# - GE removed (legacy Eurodollar)
# Uses Databento continuous front-month symbology (.v.0)
SYMBOLS = [
    # === A) GRAINS & OILSEEDS ===
    "ZL",
    "ZS",
    "ZM",  # Soy Complex
    "ZC",
    "ZW",
    "KE",
    "ZO",  # Corn/Wheat/Oats
    "ZR",
    "XC",
    "XW",
    "XK",  # Mini Grains
    "CPO",  # Crude Palm Oil (CME USD Cash-Settled)
    # === B) LIVESTOCK & DAIRY ===
    "LE",
    "GF",
    "HE",  # Cattle/Hogs
    "DC",
    "DY",
    # === C) SOFTS (CME/NYMEX ALTERNATIVES) ===
    "YO",  # Sugar No. 11 (Cash Settled)
    "KT",  # Coffee (Cash Settled)
    "CJ",  # Cocoa (Cash Settled)
    "TT",  # Cotton (Cash Settled)
    "LBR",  # Lumber (Physically Delivered)
    # === D) ENERGY ===
    "CL",
    "BZ",  # Crude (WTI/Brent)
    "HO",
    "RB",
    "NG",  # Heating Oil/RBOB/NatGas
    "QM",
    "QG",
    "QH",
    "QU",  # E-minis
    "MCL",  # Micro Crude
    "CU",  # Chicago Ethanol (Platts)
    # === E) METALS ===
    "GC",
    "SI",
    "HG",  # Gold/Silver/Copper
    "PL",
    "PA",
    "ALI",  # Platinum/Palladium/Aluminum
    "MGC",
    "SIL",  # Micro Silver (CME root)
    "QI",
    "QO",  # Mini Silver/Gold
    # === F) EQUITY INDICES ===
    "ES",
    "NQ",
    "YM",
    "RTY",  # E-minis (S&P, Nas, Dow, Russ)
    "EMD",  # MidCap 400
    "MES",
    "MNQ",
    "MYM",
    "M2K",  # Micro Indices
    "NIY",  # Nikkei 225 (USD)
    # === G) FX (FUTURES ONLY) ===
    "6E",
    "6A",
    "6J",
    "6B",  # Euro, Aussie, Yen, Pound
    "6C",
    "6N",
    "6S",
    "6M",  # CAD, Kiwi, Swiss, Peso
    "6L",
    "6Z",
    "6R",  # BRL, Rand, Ruble
    "M6E",
    "M6A",
    "M6B",  # Micro FX
    # === H) INTEREST RATES & YIELDS ===
    "ZQ",  # 30-Day Fed Funds
    "ZT",
    "ZF",
    "ZN",  # 2Y, 5Y, 10Y Treasury Notes
    "TN",
    "ZB",
    "UB",  # Ultra 10Y, 30Y Bond, Ultra Bond
    "SR1",
    "SR3",  # 1M / 3M SOFR
    # Yield Futures (Financial)
    "2YY",
    "5YY",
    "10Y",
    "30Y",
    # === I) CRYPTO ===
    "BTC",
    "ETH",  # Bitcoin/Ether
    "MBT",
    "MET",  # Micro Crypto
]

# NOTE: VX (VIX) excluded - trades on CBOE, not CME GLBX.MDP3
# NOTE: ICE originals (KC/SB/CC/CT) excluded - use CME alternatives (KT/YO/CJ/TT)
# NOTE: FCPO (Bursa) excluded - use CPO (CME cash-settled palm oil) instead


def get_connection():
    """Get MotherDuck connection (CLOUD ONLY - no local fallback)"""
    if not MOTHERDUCK_TOKEN:
        raise RuntimeError("MOTHERDUCK_TOKEN not set - cannot proceed (CLOUD ONLY)")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")


def get_databento_client():
    """Initialize Databento client - uses environment variable only"""
    try:
        import databento as db

        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY environment variable not set")
        client = db.Historical(key=api_key)
        logger.info("Initialized Databento client successfully")
        return client
    except ImportError:
        logger.error(
            "databento package not installed. Install with: pip install databento"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Databento client: {e}")
        return None


def estimate_cost(client, symbols: list, start_date: str, end_date: str) -> float:
    """
    Estimate cost of data request before executing.
    Returns cost in USD.
    """
    try:
        # Build continuous symbols
        continuous_symbols = [f"{s}.v.0" for s in symbols]

        cost_response = client.metadata.get_cost(
            dataset=DATASET,
            symbols=continuous_symbols,
            schema="ohlcv-1d",
            start=start_date,
            end=end_date,
            stype_in="continuous",
        )
        return cost_response
    except Exception as e:
        logger.warning(f"Could not estimate cost: {e}")
        return -1


def collect_batch_data(
    client,
    symbols: list,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Collect daily OHLCV data for MULTIPLE symbols in ONE request.

    Per Databento docs: Up to 2,000 symbols per request.
    This is much more efficient than individual requests.

    Args:
        client: Databento Historical client
        symbols: List of futures root symbols (e.g., ['ZL', 'ZS', 'ZC'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_retries: Max retry attempts with exponential backoff

    Returns:
        DataFrame with columns: symbol, as_of_date, open, high, low, close, volume, open_interest
    """
    # Build continuous symbols (.v.0 = volume-based front month)
    continuous_symbols = [f"{s}.v.0" for s in symbols]

    logger.info(f"Requesting {len(symbols)} symbols from {start_date} to {end_date}")
    logger.info(f"Symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")

    for attempt in range(max_retries):
        try:
            # Single batched request for ALL symbols
            data = client.timeseries.get_range(
                dataset=DATASET,
                symbols=continuous_symbols,
                schema="ohlcv-1d",
                start=start_date,
                end=end_date,
                stype_in="continuous",  # Required for .v.0 symbology
            )

            # Convert to DataFrame
            df = data.to_df().reset_index()

            if df.empty:
                logger.warning("No data returned from Databento")
                return pd.DataFrame()

            # Process the data
            df = df.rename(columns={"ts_event": "as_of_date"})

            # Extract root symbol from continuous symbol (ZL.v.0 -> ZL)
            df["symbol"] = df["symbol"].str.replace(r"\.v\.\d+$", "", regex=True)

            # Ensure as_of_date is DATE type
            df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date

            # open_interest not available in ohlcv-1d schema - set to NULL
            if "open_interest" not in df.columns:
                df["open_interest"] = None

            # Select columns in order (matching DDL schema)
            df = df[
                [
                    "symbol",
                    "as_of_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "open_interest",
                ]
            ]

            logger.info(f"Received {len(df)} rows for {df['symbol'].nunique()} symbols")
            return df

        except Exception as e:
            error_str = str(e)

            # Handle rate limiting (429)
            if "429" in error_str or "rate" in error_str.lower():
                wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2, 4, 8 seconds
                logger.warning(
                    f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}"
                )
                time.sleep(wait_time)
                continue

            # Handle other errors
            logger.error(
                f"Error collecting data (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                raise

    return pd.DataFrame()


def main(
    dry_run: bool = False,
    start_date_override: str | None = None,
    symbols_override: list[str] | None = None,
):
    """
    Main ingestion function (CLOUD ONLY - writes to MotherDuck).

    Args:
        dry_run: If True, estimate cost but don't fetch data
    """
    logger.info("=" * 60)
    logger.info("DATABENTO HISTORICAL DATA COLLECTION")
    logger.info("=" * 60)

    client = get_databento_client()
    if not client:
        return

    con = get_connection()
    logger.info("Target: MotherDuck (CLOUD ONLY)")
    logger.info(f"Dataset: {DATASET}")
    symbols = symbols_override or SYMBOLS
    logger.info(f"Symbols: {len(symbols)}")

    # Determine start date (resume from last data or start fresh)
    try:
        if start_date_override:
            start_date = pd.to_datetime(start_date_override).date().strftime("%Y-%m-%d")
            logger.info(f"Start date overridden to {start_date}")
        else:
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
                logger.info(f"Resuming from {start_date} (last data: {last_date})")
            else:
                start_date = DATASET_START
                logger.info(f"No existing data. Starting from {start_date}")
    except Exception as e:
        logger.warning(f"Could not check existing data: {e}")
        start_date = DATASET_START

    end_date = datetime.now().strftime("%Y-%m-%d")

    # Skip if we're already up to date
    if start_date >= end_date:
        logger.info("Data is already up to date!")
        con.close()
        return

    # Estimate cost
    logger.info("Estimating cost...")
    estimated_cost = estimate_cost(client, SYMBOLS, start_date, end_date)
    if estimated_cost > 0:
        logger.info(f"Estimated cost: ${estimated_cost:.2f}")

    if dry_run:
        logger.info("DRY RUN - skipping data fetch")
        con.close()
        return

    # Collect data (single batched request)
    logger.info("Fetching data...")
    df = collect_batch_data(client, symbols, start_date, end_date)

    if df.empty:
        logger.warning("No data collected")
        con.close()
        return

    logger.info(f"Total rows collected: {len(df):,}")
    logger.info(f"Symbols with data: {df['symbol'].nunique()}")
    logger.info(f"Date range: {df['as_of_date'].min()} to {df['as_of_date'].max()}")

    # Load to MotherDuck with idempotent upsert (DELETE + INSERT pattern)
    logger.info("Loading to database...")
    try:
        # Register DataFrame and use DELETE + INSERT for clean upsert
        # (INSERT OR REPLACE doesn't work well with default columns in DuckDB)
        con.register("staging_df", df)

        # Delete existing rows that will be replaced
        con.execute(
            """
            DELETE FROM raw.databento_futures_ohlcv_1d
            WHERE (symbol, as_of_date) IN (
                SELECT symbol, as_of_date FROM staging_df
            )
        """
        )

        # Insert new data (source and ingested_at use DDL defaults)
        con.execute(
            """
            INSERT INTO raw.databento_futures_ohlcv_1d 
                (symbol, as_of_date, open, high, low, close, volume, open_interest)
            SELECT symbol, as_of_date, open, high, low, close, volume, open_interest
            FROM staging_df
        """
        )

        logger.info(
            f"Successfully loaded {len(df):,} rows to raw.databento_futures_ohlcv_1d"
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    finally:
        con.close()

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect Databento historical OHLCV data (CLOUD ONLY)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Estimate cost without fetching data"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override start date (YYYY-MM-DD) for backfill",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional list of symbols to fetch (defaults to full SYMBOLS list)",
    )

    args = parser.parse_args()
    main(
        dry_run=args.dry_run,
        start_date_override=args.start_date,
        symbols_override=args.symbols,
    )
