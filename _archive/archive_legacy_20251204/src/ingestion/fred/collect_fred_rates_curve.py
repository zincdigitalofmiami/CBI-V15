#!/usr/bin/env python3
"""
FRED Rates & Curve Ingestion - Mac-only, manual run

Bucket: fred_rates_curve

Pulls key interest rate and curve-related FRED series into:
  - raw.fred_economic (date, series_id, value)
via staging tables:
  - raw_staging.fred_rates_curve_<run_id>
"""

import sys
from pathlib import Path
from datetime import date, datetime
from typing import List

import logging
import requests
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cbi_utils.keychain_manager import get_api_key  # noqa: E402
from src.cbi_utils.bigquery_client import (  # noqa: E402
    get_client,
    load_dataframe_to_table,
    merge_staging_to_target,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
TABLE_ID = f"{PROJECT_ID}.raw.fred_economic"

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

RATES_SERIES: List[str] = [
    # Policy & overnight
    "DFF",        # Effective Fed Funds
    "DFEDTARU",   # Fed funds target range upper
    "DFEDTARL",   # Fed funds target range lower
    "EFFR",       # Effective Fed Funds Rate
    "SOFR",       # Secured Overnight Financing Rate
    # Treasury curve
    "DGS3MO",     # 3-month
    "DGS1",       # 1-year
    "DGS2",       # 2-year
    "DGS5",       # 5-year
    "DGS10",      # 10-year
    "DGS30",      # 30-year
    # Spreads
    "T10Y2Y",     # 10Y-2Y spread
    "T10Y3M",     # 10Y-3M spread
]


def get_last_loaded_date() -> date:
    """
    Get the last loaded date across the rates/curve series in raw.fred_economic.
    If none found, return a default backfill start date.
    """
    client = get_client()
    series_list = ",".join(f"'{s}'" for s in RATES_SERIES)
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
            logger.info(f"Last rates/curve date in {TABLE_ID}: {last_dt}")
            return last_dt
    except Exception as e:
        logger.warning(f"Could not determine last rates/curve date from {TABLE_ID}: {e}")

    default_start = date(2005, 1, 1)
    logger.info(f"No existing rates/curve data found; starting from {default_start}")
    return default_start


def fetch_fred_series(series_id: str, api_key: str, start_dt: date, end_dt: date) -> pd.DataFrame:
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

    if start_dt >= end_dt:
        logger.info("Rates/curve series already up to date; nothing to fetch.")
        return

    all_frames = []
    for sid in RATES_SERIES:
        df = fetch_fred_series(sid, api_key, start_dt, end_dt)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.warning("No rates/curve data fetched from FRED.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["series_id", "date"]).drop_duplicates(["series_id", "date"])
    logger.info(f"Total rates/curve rows fetched this run: {len(combined)}")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    staging_dataset = f"{PROJECT_ID}.raw_staging"
    staging_table = f"{staging_dataset}.fred_rates_curve_{run_id}"

    client = get_client(project_id=PROJECT_ID)
    try:
        client.get_dataset(staging_dataset)
    except Exception:
        from google.cloud import bigquery as bq
        client.create_dataset(bq.Dataset(staging_dataset))

    if not load_dataframe_to_table(combined, staging_table, PROJECT_ID, "WRITE_TRUNCATE"):
        logger.error(f"❌ Failed to load rates/curve series into staging table {staging_table}")
        return

    if merge_staging_to_target(
        staging_table=staging_table,
        target_table=TABLE_ID,
        key_columns=["series_id", "date"],
        all_columns=["date", "series_id", "value"],
        project_id=PROJECT_ID,
    ):
        logger.info(f"✅ Merged rates/curve series from {staging_table} into {TABLE_ID}")
    else:
        logger.error(f"❌ MERGE from {staging_table} into {TABLE_ID} failed")


if __name__ == "__main__":
    main()

