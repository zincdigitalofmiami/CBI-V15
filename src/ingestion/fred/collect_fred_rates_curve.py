#!/usr/bin/env python3
"""
FRED Rates Curve Ingestion - MotherDuck

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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR / ".env.local", override=True)
except ImportError:
    pass

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
PARQUET_DIR = ROOT_DIR / "data" / "raw" / "fred"

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
    """Get MotherDuck connection (CLOUD ONLY - no local fallback)"""
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set - cannot proceed (CLOUD ONLY)")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={token}")


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
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
    con.register("combined", combined)
    con.execute(
        """
        INSERT OR REPLACE INTO raw.fred_economic (date, series_id, value)
        SELECT date, series_id, value FROM combined
    """
    )
    logger.info(f"✅ Upserted {len(combined)} rows")


if __name__ == "__main__":
    main()
