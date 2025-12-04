#!/usr/bin/env python3
"""
FRED Financial Conditions Ingestion - MotherDuck/DuckDB

Bucket: fred_financial

Uses FRED as source for financial conditions indicators:
  - NFCI       (Chicago Fed National Financial Conditions Index)
  - STLFSI4    (St. Louis Fed Financial Stress Index)
  - TEDRATE    (TED Spread)
  - T10Y2Y     (10Y-2Y Treasury Spread)
  - T10Y3M     (10Y-3M Treasury Spread)

Writes to: raw.fred_economic (date, series_id, value)
"""

import os
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List

import duckdb
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")
PARQUET_DIR = Path("/Volumes/Satechi Hub/CBI-V15/data/raw/fred")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FINANCIAL_SERIES: List[str] = [
    "NFCI",
    "STLFSI4",
    "TEDRATE",
    "T10Y2Y",
    "T10Y3M",
]


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get MotherDuck connection."""
    return duckdb.connect(f"md:{MOTHERDUCK_DB}")


def get_last_loaded_date(con: duckdb.DuckDBPyConnection) -> date:
    """Get the last loaded date across the financial series."""
    series_list = ",".join(f"'{s}'" for s in FINANCIAL_SERIES)
    query = f"""
    SELECT MAX(date) AS last_date
    FROM raw.fred_economic
    WHERE series_id IN ({series_list})
    """
    try:
        result = con.execute(query).fetchone()
        if result and result[0]:
            return result[0]
    except Exception as e:
        logger.warning(f"Could not determine last date: {e}")

    return date(2005, 1, 1)


def fetch_fred_series(
    series_id: str, api_key: str, start_dt: date, end_dt: date
) -> pd.DataFrame:
    """Fetch a single FRED series."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_dt.isoformat(),
        "observation_end": end_dt.isoformat(),
    }
    logger.info(f"Fetching {series_id} from {start_dt} to {end_dt}")
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get("observations", [])
    rows = []
    for obs in observations:
        v = obs.get("value")
        if v is None or v == ".":
            continue
        try:
            rows.append(
                {
                    "date": pd.to_datetime(obs["date"]).date(),
                    "series_id": series_id,
                    "value": float(v),
                }
            )
        except ValueError:
            continue

    df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["date", "series_id", "value"])
    )
    logger.info(f"Fetched {len(df)} rows for {series_id}")
    return df


def main():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")

    con = get_connection()
    last_date = get_last_loaded_date(con)
    today = datetime.utcnow().date()

    if last_date >= today:
        logger.info("Financial conditions data up to date.")
        return

    all_frames = []
    for sid in FINANCIAL_SERIES:
        df = fetch_fred_series(sid, api_key, last_date, today)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No data fetched.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(["series_id", "date"])
    logger.info(f"Total rows: {len(combined)}")

    # Save to Parquet
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = (
        PARQUET_DIR / f"fred_financial_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    )
    combined.to_parquet(parquet_path, index=False)

    # Upsert to MotherDuck
    con.execute(
        """
        INSERT OR REPLACE INTO raw.fred_economic (date, series_id, value)
        SELECT date, series_id, value FROM combined
    """
    )
    logger.info(f"✅ Upserted {len(combined)} rows")


if __name__ == "__main__":
    main()
