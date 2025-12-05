#!/usr/bin/env python3
"""
Collect Databento statistics messages for a set of futures symbols.

Writes per-event statistics into:
  raw.databento_futures_statistics
via raw_staging.databento_statistics_<run_id> and MERGE on:
  (symbol, ts_event, stat_type)

This does NOT aggregate; it stores the raw StatMsg stream so that
daily/hourly stats can be derived later in feature builders.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

import logging
import pandas as pd

from google.cloud import bigquery

# Project root for utils import
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cbi_utils.keychain_manager import get_api_key  # noqa: E402
from src.cbi_utils.bigquery_client import (  # noqa: E402
    get_client,
    load_dataframe_to_table,
    merge_staging_to_target,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
RAW_TABLE = f"{PROJECT_ID}.raw.databento_futures_statistics"
STAGING_DATASET = f"{PROJECT_ID}.raw_staging"

# Use the same symbol universe as pull_missing_symbols.py
SYMBOLS: List[str] = [
    # Agricultural (CBOT)
    "ZL.FUT",
    "ZS.FUT",
    "ZM.FUT",
    "ZC.FUT",
    "ZW.FUT",
    "ZO.FUT",
    "KC.FUT",
    "CT.FUT",
    "SB.FUT",
    "LH.FUT",
    "LC.FUT",
    "FC.FUT",
    # Energy (NYMEX)
    "CL.FUT",
    "HO.FUT",
    "RB.FUT",
    "NG.FUT",
    # Metals (COMEX/NYMEX)
    "GC.FUT",
    "SI.FUT",
    "HG.FUT",
    "PL.FUT",
    "PA.FUT",
    # Financials / FX (CME)
    "ES.FUT",
    "NQ.FUT",
    "YM.FUT",
    "RTY.FUT",
    "ZN.FUT",
    "ZB.FUT",
    "ZF.FUT",
    "ZT.FUT",
    "6E.FUT",
    "6B.FUT",
    "6J.FUT",
    "6A.FUT",
    "6C.FUT",
    "6S.FUT",
    "6N.FUT",
]


def get_databento_client():
    """Initialize Databento client."""
    try:
        import databento as db  # type: ignore
    except ImportError:
        logger.error("databento package not installed. Install with: pip install databento")
        return None

    api_key = get_api_key("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not found in Keychain/env.")
        return None

    try:
        client = db.Historical(key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Databento client: {e}")
        return None


def get_last_ts_for_symbol(symbol: str) -> datetime | None:
    """
    Return the latest ts_event we have for a given root symbol in RAW_TABLE.
    """
    client = get_client(project_id=PROJECT_ID)
    query = f"""
    SELECT MAX(ts_event) AS last_ts
    FROM `{RAW_TABLE}`
    WHERE symbol = @symbol
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("symbol", "STRING", symbol)]
    )
    try:
        df = client.query(query, job_config=job_config).to_dataframe()
        if df.empty:
            return None
        last = df["last_ts"].iloc[0]
        if pd.notna(last):
            return pd.to_datetime(last).to_pydatetime()
    except Exception as e:
        logger.warning(f"Could not determine last ts_event for {symbol}: {e}")
    return None


def collect_statistics_for_symbol(symbol_fut: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Collect Databento statistics messages for one symbol between start_dt and end_dt.
    """
    client = get_databento_client()
    if not client:
        return pd.DataFrame()

    try:
        logger.info(f"Collecting statistics for {symbol_fut} from {start_dt} to {end_dt}")
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol_fut],
            stype_in="parent",
            schema="statistics",
            start=start_dt,
            end=end_dt,
        )
    except Exception as e:
        logger.error(f"Error requesting statistics for {symbol_fut}: {e}")
        return pd.DataFrame()

    records = []
    root_symbol = symbol_fut.replace(".FUT", "")
    try:
        for rec in store:
            # Use pretty_ts_event for wall-clock time
            ts = pd.to_datetime(rec.pretty_ts_event)
            records.append(
                {
                    "date": ts.date(),
                    "ts_event": ts.to_pydatetime(),
                    "symbol": root_symbol,
                    "price": float(rec.pretty_price),
                    "quantity": int(rec.quantity),
                    "stat_type": int(rec.stat_type),
                    "stat_flags": int(rec.stat_flags),
                }
            )
    except Exception as e:
        logger.error(f"Error iterating statistics for {symbol_fut}: {e}")
        return pd.DataFrame()

    if not records:
        logger.info(f"No statistics records for {symbol_fut} in requested range.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    df["symbol"] = df["symbol"].astype(str)
    df["price"] = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    df["stat_type"] = df["stat_type"].astype(int)
    df["stat_flags"] = df["stat_flags"].astype(int)

    # Drop exact duplicates
    # Map to existing BigQuery schema:
    # date, symbol, stat_type (STRING), price, quantity, ts_event, ts_recv, update_action
    df["stat_type"] = df["stat_type"].astype(str)
    # For now, reuse ts_event as ts_recv and simple update_action flag
    df["ts_recv"] = df["ts_event"]
    df["update_action"] = "U"

    df = df[["date", "symbol", "stat_type", "price", "quantity", "ts_event", "ts_recv", "update_action"]]
    df = df.drop_duplicates(subset=["symbol", "ts_event", "stat_type", "update_action"], keep="last")
    logger.info(f"Collected {len(df)} statistics rows for {symbol_fut}")
    return df


def main(days_back: int = 1) -> None:
    """
    Collect statistics for all configured symbols for the recent time window.
    """
    client = get_client(project_id=PROJECT_ID)
    # Ensure raw_staging dataset exists
    try:
        client.get_dataset(STAGING_DATASET)
    except Exception:
        client.create_dataset(bigquery.Dataset(STAGING_DATASET))

    end_dt = datetime.utcnow()
    start_dt_default = end_dt - timedelta(days=days_back)

    all_frames: List[pd.DataFrame] = []

    for symbol_fut in SYMBOLS:
        root_symbol = symbol_fut.replace(".FUT", "")
        last_ts = get_last_ts_for_symbol(root_symbol)
        if last_ts is not None and last_ts < end_dt:
            start_dt = last_ts + timedelta(microseconds=1)
        else:
            start_dt = start_dt_default

        if start_dt >= end_dt:
            logger.info(f"{root_symbol}: statistics already up to date.")
            continue

        df = collect_statistics_for_symbol(symbol_fut, start_dt, end_dt)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.info("No new statistics data to load.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Total new statistics rows this run: {len(combined)}")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    staging_table = f"{STAGING_DATASET}.databento_statistics_{run_id}"

    if not load_dataframe_to_table(combined, staging_table, PROJECT_ID, "WRITE_TRUNCATE"):
        logger.error(f"❌ Failed to load statistics into staging table {staging_table}")
        return

    if merge_staging_to_target(
        staging_table=staging_table,
        target_table=RAW_TABLE,
        key_columns=["symbol", "ts_event", "stat_type", "update_action"],
        all_columns=["date", "symbol", "stat_type", "price", "quantity", "ts_event", "ts_recv", "update_action"],
        project_id=PROJECT_ID,
    ):
        logger.info(f"✅ MERGE from {staging_table} into {RAW_TABLE} complete")
    else:
        logger.error(f"❌ MERGE from {staging_table} into {RAW_TABLE} failed")


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(days_back=days)
