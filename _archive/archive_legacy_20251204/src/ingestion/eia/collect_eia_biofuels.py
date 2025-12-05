#!/usr/bin/env python3
"""
EIA Biofuels Ingestion - Bucket: eia_biofuels

Role:
- Pull biofuel / RIN-related series from the EIA API.
- Write them into raw.eia_biofuels (date, series_id, value) via
  raw_staging.eia_biofuels_<run_id> and a MERGE on (series_id, date).

Notes:
- This script is architecture-compliant but requires:
  - An EIA API key stored under "EIA_API_KEY" (Keychain/env/Secret Manager).
  - Actual EIA series IDs configured via environment variables (see SERIES_ENV_MAP).
- No fake data: if no real series IDs are configured, the script raises.
"""

import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import logging
import requests
import pandas as pd

from google.cloud import bigquery

# Project root for utils import
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cbi_utils.keychain_manager import get_api_key  # noqa: E402
from src.cbi_utils.bigquery_client import (  # noqa: E402
    get_client,
    load_dataframe_to_table,
    merge_staging_to_target,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GCP_PROJECT", "cbi-v15")
RAW_TABLE = f"{PROJECT_ID}.raw.eia_biofuels"
STAGING_DATASET = f"{PROJECT_ID}.raw_staging"
EIA_BASE_URL = "https://api.eia.gov/series/"

# Map from logical bucket names to environment variables that hold the real EIA series IDs.
# You must set these env vars (or secrets) to valid EIA series IDs before running.
SERIES_ENV_MAP: Dict[str, str] = {
    "rin_prices_d4": "EIA_SERIES_RIN_D4",
    "rin_prices_d6": "EIA_SERIES_RIN_D6",
    "biodiesel_production": "EIA_SERIES_BIODIESEL_PROD",
    "rfs_volumes": "EIA_SERIES_RFS_VOLUMES",
}


def resolve_series_ids_from_env() -> Dict[str, str]:
    """
    Resolve logical series aliases to actual EIA series IDs via environment variables.

    Returns:
        dict mapping alias -> EIA series_id
    """
    resolved: Dict[str, str] = {}
    for alias, env_var in SERIES_ENV_MAP.items():
        sid = os.getenv(env_var)
        if sid:
            resolved[alias] = sid
        else:
            logger.info(f"[eia_biofuels] Env var {env_var} not set; skipping alias '{alias}'")
    return resolved


def get_last_loaded_date_for_series(series_id: str) -> Optional[date]:
    """
    Get the last loaded date for a given EIA series in raw.eia_biofuels.
    Returns None if no data exists yet.
    """
    client = get_client(project_id=PROJECT_ID)
    query = f"""
    SELECT MAX(date) AS last_date
    FROM `{RAW_TABLE}`
    WHERE series_id = @series_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("series_id", "STRING", series_id)]
    )
    try:
        df = client.query(query, job_config=job_config).to_dataframe()
        if df.empty:
            return None
        last = df["last_date"].iloc[0]
        if pd.notna(last):
            return pd.to_datetime(last).date()
    except Exception as e:
        logger.warning(f"[eia_biofuels] Could not determine last date for {series_id}: {e}")
    return None


def parse_eia_date(date_str: str) -> date:
    """
    Parse EIA series date strings into a Python date.
    Handles common EIA formats: YYYY, YYYYMM, YYYYMMDD, YYYY-MM-DD.
    """
    ds = date_str.strip()
    # Common formats
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(ds, fmt).date()
        except ValueError:
            pass

    # Monthly (YYYYMM) -> first of month
    if len(ds) == 6:
        return datetime.strptime(ds, "%Y%m").date().replace(day=1)

    # Yearly (YYYY) -> Jan 1
    if len(ds) == 4:
        return datetime.strptime(ds, "%Y").date().replace(month=1, day=1)

    # Fallback: try ISO parser
    return pd.to_datetime(ds).date()


def fetch_eia_series(series_id: str, api_key: str) -> pd.DataFrame:
    """
    Fetch a single EIA series and return as DataFrame with columns: date, series_id, value.
    """
    params = {
        "api_key": api_key,
        "series_id": series_id,
    }
    logger.info(f"[eia_biofuels] Fetching EIA series {series_id}")
    resp = requests.get(EIA_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    series_list = data.get("series", [])
    if not series_list:
        logger.warning(f"[eia_biofuels] No 'series' field in EIA response for {series_id}")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    series_obj = series_list[0]
    observations = series_obj.get("data", [])
    if not observations:
        logger.warning(f"[eia_biofuels] No data points for series {series_id}")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    rows: List[Tuple[date, str, float]] = []
    for obs in observations:
        # EIA returns [date_str, value] pairs
        if not isinstance(obs, (list, tuple)) or len(obs) < 2:
            continue
        ds, val = obs[0], obs[1]
        try:
            dt = parse_eia_date(str(ds))
        except Exception:
            continue
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        rows.append((dt, series_id, fv))

    if not rows:
        logger.warning(f"[eia_biofuels] All observations for {series_id} were invalid/empty")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    df = pd.DataFrame(rows, columns=["date", "series_id", "value"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["series_id"] = df["series_id"].astype(str)
    df["value"] = df["value"].astype(float)
    logger.info(f"[eia_biofuels] Fetched {len(df)} rows for {series_id}")
    return df


def main():
    api_key = get_api_key("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("EIA_API_KEY not found in env/Keychain/Secret Manager")

    series_alias_to_id = resolve_series_ids_from_env()
    if not series_alias_to_id:
        raise RuntimeError(
            "[eia_biofuels] No EIA series IDs configured. "
            "Set env vars EIA_SERIES_RIN_D4, EIA_SERIES_RIN_D6, "
            "EIA_SERIES_BIODIESEL_PROD, EIA_SERIES_RFS_VOLUMES to real EIA series IDs."
        )

    all_frames: List[pd.DataFrame] = []

    for alias, series_id in series_alias_to_id.items():
        last_dt = get_last_loaded_date_for_series(series_id)
        df = fetch_eia_series(series_id, api_key)
        if df.empty:
            continue
        if last_dt is not None:
            df = df[df["date"] > last_dt]
            logger.info(
                f"[eia_biofuels] After filtering existing rows, {len(df)} new rows for {series_id} (alias {alias})"
            )
        all_frames.append(df)

    if not all_frames:
        logger.info("[eia_biofuels] No new EIA biofuels data to load.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["series_id", "date"]).drop_duplicates(["series_id", "date"])
    logger.info(f"[eia_biofuels] Total new rows this run: {len(combined)}")

    # Ensure staging dataset exists
    client = get_client(project_id=PROJECT_ID)
    try:
        client.get_dataset(STAGING_DATASET)
    except Exception:
        client.create_dataset(bigquery.Dataset(STAGING_DATASET))

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    staging_table = f"{STAGING_DATASET}.eia_biofuels_{run_id}"

    if not load_dataframe_to_table(combined, staging_table, PROJECT_ID, "WRITE_TRUNCATE"):
        logger.error(f"[eia_biofuels] ❌ Failed to load into staging table {staging_table}")
        return

    if merge_staging_to_target(
        staging_table=staging_table,
        target_table=RAW_TABLE,
        key_columns=["series_id", "date"],
        all_columns=["date", "series_id", "value"],
        project_id=PROJECT_ID,
    ):
        logger.info(f"[eia_biofuels] ✅ MERGE from {staging_table} into {RAW_TABLE} complete")
    else:
        logger.error(f"[eia_biofuels] ❌ MERGE from {staging_table} into {RAW_TABLE} failed")


if __name__ == "__main__":
    main()

