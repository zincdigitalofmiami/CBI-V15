#!/usr/bin/env python3
"""
FRED FX Spot Ingestion - Mac-only, manual run

Bucket: fred_fx

Uses FRED as the canonical FX spot source for:
  - DTWEXBGS   (Broad dollar index)
  - DTWEXEMEGS (EM dollar index)
  - DTWEXAFEGS (Advanced economies dollar index)
  - DEXBZUS    (BRL/USD)
  - DEXMXUS    (MXN/USD)
  - DEXCHUS    (CHF/USD)
  - DEXUSEU    (USD/EUR)

Writes to: cbi-v15.raw.fred_economic (date, series_id, value)
via staging tables: cbi-v15.raw_staging.fred_fx_<run_id>
"""

import sys
from pathlib import Path
from datetime import date, datetime
from typing import List

import logging
import requests
import pandas as pd

# Project root for utils import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cbi_utils.keychain_manager import get_api_key
from src.cbi_utils.bigquery_client import get_client, load_dataframe_to_table, merge_staging_to_target


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
TABLE_ID = f"{PROJECT_ID}.raw.fred_economic"

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


def get_last_loaded_date() -> date:
    """
    Get the last loaded date across the FX series in raw.fred_economic.
    If none found, return a default backfill start date.
    """
    client = get_client()
    series_list = ",".join(f"'{s}'" for s in FX_SERIES)
    query = f"""
    SELECT MAX(date) AS last_date
    FROM `{TABLE_ID}`
    WHERE series_id IN ({series_list})
    """
    try:
        df = client.query(query).to_dataframe()
        last = df["last_date"].iloc[0]
        if pd.notna(last):
            last_dt = pd.to_datetime(last).date()
            logger.info(f"Last FX date in {TABLE_ID}: {last_dt}")
            return last_dt
    except Exception as e:
        logger.warning(f"Could not determine last FX date from {TABLE_ID}: {e}")

    # Default start date if table/series is empty
    default_start = date(2005, 1, 1)
    logger.info(f"No existing FX data found; starting from {default_start}")
    return default_start


def fetch_fred_series(series_id: str, api_key: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    """
    Fetch a single FRED series between start_dt and end_dt.
    Returns a DataFrame with columns: date, series_id, value
    """
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
    api_key = get_api_key("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not found in env/Keychain/Secret Manager")

    last_date = get_last_loaded_date()
    start_dt = last_date
    today = datetime.utcnow().date()
    end_dt = today

    # If last_date is today or later, nothing to do
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
    combined = combined.sort_values(["series_id", "date"]).drop_duplicates(["series_id", "date"])
    logger.info(f"Total FX rows fetched this run: {len(combined)}")

    # Load into a run-scoped staging table, then MERGE into canonical raw.fred_economic
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    staging_dataset = f"{PROJECT_ID}.raw_staging"
    staging_table = f"{staging_dataset}.fred_fx_{run_id}"

    # Ensure staging dataset exists
    client = get_client(project_id=PROJECT_ID)
    try:
        client.get_dataset(staging_dataset)
    except Exception:
        from google.cloud import bigquery as bq
        client.create_dataset(bq.Dataset(staging_dataset))

    if not load_dataframe_to_table(combined, staging_table, PROJECT_ID, "WRITE_TRUNCATE"):
        logger.error(f"❌ Failed to load FX series into staging table {staging_table}")
        return

    if merge_staging_to_target(
        staging_table=staging_table,
        target_table=TABLE_ID,
        key_columns=["series_id", "date"],
        all_columns=["date", "series_id", "value"],
        project_id=PROJECT_ID,
    ):
        logger.info(f"✅ Merged FX series from {staging_table} into {TABLE_ID}")
    else:
        logger.error(f"❌ MERGE from {staging_table} into {TABLE_ID} failed")


if __name__ == "__main__":
    main()
