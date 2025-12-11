#!/usr/bin/env python3
"""
FRED FX Spot Ingestion - MotherDuck/DuckDB

Bucket: fred_fx

Uses FRED as the canonical FX spot source for:
  - DTWEXBGS   (Broad dollar index)
  - DTWEXEMEGS (EM dollar index)
  - DTWEXAFEGS (Advanced economies dollar index)
  - DEXBZUS    (BRL/USD)
  - DEXMXUS    (MXN/USD)
  - DEXCHUS    (CHF/USD)
  - DEXUSEU    (USD/EUR)

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

# MotherDuck connection
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")
PARQUET_DIR = Path("/Volumes/Satechi Hub/CBI-V15/data/raw/fred")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FX_SERIES: List[str] = [
    "DTWEXBGS",
    "DTWEXEMEGS",
    "DTWEXAFEGS",
    "DEXBZUS",
    "DEXMXUS",
    "DEXCHUS",
    "DEXUSEU",
]


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get MotherDuck connection."""
    return duckdb.connect(f"md:{MOTHERDUCK_DB}")


def get_last_loaded_date(con: duckdb.DuckDBPyConnection) -> date:
    """Get the last loaded date across the FX series in raw.fred_economic."""
    series_list = ",".join(f"'{s}'" for s in FX_SERIES)
    query = f"""
    SELECT MAX(date) AS last_date
    FROM raw.fred_economic
    WHERE series_id IN ({series_list})
    """
    try:
        result = con.execute(query).fetchone()
        if result and result[0]:
            last_dt = result[0]
            logger.info(f"Last FX date in raw.fred_economic: {last_dt}")
            return last_dt
    except Exception as e:
        logger.warning(f"Could not determine last FX date: {e}")

    # Default start date if table/series is empty
    default_start = date(2005, 1, 1)
    logger.info(f"No existing FX data found; starting from {default_start}")
    return default_start


def fetch_fred_series(
    series_id: str, api_key: str, start_dt: date, end_dt: date
) -> pd.DataFrame:
    """Fetch a single FRED series between start_dt and end_dt."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_dt.isoformat(),
        "observation_end": end_dt.isoformat(),
    }
    logger.info(f"Fetching FRED series {series_id} from {start_dt} to {end_dt}")
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get("observations", [])
    if not observations:
        logger.warning(f"No observations returned for {series_id}")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    rows = []
    for obs in observations:
        v = obs.get("value")
        if v is None or v == ".":
            continue
        obs_date = pd.to_datetime(obs["date"]).date()
        try:
            val = float(v)
        except ValueError:
            continue
        rows.append({"date": obs_date, "series_id": series_id, "value": val})

    if not rows:
        logger.warning(f"All observations for {series_id} were missing/invalid")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["series_id"] = df["series_id"].astype(str)
    df["value"] = df["value"].astype(float)
    logger.info(f"Fetched {len(df)} rows for {series_id}")
    return df


def main():
    # Get API key from environment
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in environment")

    con = get_connection()

    last_date = get_last_loaded_date(con)
    start_dt = last_date
    today = datetime.utcnow().date()
    end_dt = today

    if start_dt >= end_dt:
        logger.info("FX series already up to date; nothing to fetch.")
        return

    all_frames = []
    for sid in FX_SERIES:
        df = fetch_fred_series(sid, api_key, start_dt, end_dt)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No FX data fetched from FRED.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["series_id", "date"]).drop_duplicates(
        ["series_id", "date"]
    )
    logger.info(f"Total FX rows fetched this run: {len(combined)}")

    # Save to Parquet (data lake)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = (
        PARQUET_DIR / f"fred_fx_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    )
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"Saved to {parquet_path}")

    # Upsert to MotherDuck
    con.execute(
        """
        INSERT OR REPLACE INTO raw.fred_economic (date, series_id, value)
        SELECT date, series_id, value FROM combined
    """
    )
    logger.info(f"✅ Upserted {len(combined)} rows to raw.fred_economic")


if __name__ == "__main__":
    main()
