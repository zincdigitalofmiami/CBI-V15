#!/usr/bin/env python3
"""
FRED Rates Curve Ingestion - MotherDuck/DuckDB

Bucket: fred_rates

Uses FRED for Treasury yield curve data:
  - DGS1MO, DGS3MO, DGS6MO (short-term)
  - DGS1, DGS2, DGS5 (medium-term)
  - DGS7, DGS10, DGS20, DGS30 (long-term)

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

RATES_SERIES: List[str] = [
    "DGS1MO",
    "DGS3MO",
    "DGS6MO",
    "DGS1",
    "DGS2",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS20",
    "DGS30",
]


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get MotherDuck connection."""
    return duckdb.connect(f"md:{MOTHERDUCK_DB}")


def get_last_loaded_date(con: duckdb.DuckDBPyConnection) -> date:
    """Get the last loaded date for rates series."""
    series_list = ",".join(f"'{s}'" for s in RATES_SERIES)
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
    logger.info(f"Fetching {series_id}")
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

    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["date", "series_id", "value"])
    )


def main():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")

    con = get_connection()
    last_date = get_last_loaded_date(con)
    today = datetime.utcnow().date()

    if last_date >= today:
        logger.info("Rates data up to date.")
        return

    all_frames = []
    for sid in RATES_SERIES:
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
        PARQUET_DIR / f"fred_rates_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
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
