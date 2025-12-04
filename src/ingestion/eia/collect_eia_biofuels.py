#!/usr/bin/env python3
"""
EIA Biofuels Ingestion - MotherDuck/DuckDB

Bucket: eia_biofuels

Role:
- Pull biofuel / RIN-related series from the EIA API
- Write to raw.eia_biofuels (date, series_id, value)
- Save to Parquet data lake

Series (set via environment):
  - EIA_SERIES_RIN_D4
  - EIA_SERIES_RIN_D6
  - EIA_SERIES_BIODIESEL_PROD
  - EIA_SERIES_RFS_VOLUMES
"""

import os
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import duckdb
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")
PARQUET_DIR = Path("/Volumes/Satechi Hub/CBI-V15/data/raw/eia")
EIA_BASE_URL = "https://api.eia.gov/series/"

# Map from logical bucket names to environment variables that hold EIA series IDs
SERIES_ENV_MAP: Dict[str, str] = {
    "rin_prices_d4": "EIA_SERIES_RIN_D4",
    "rin_prices_d6": "EIA_SERIES_RIN_D6",
    "biodiesel_production": "EIA_SERIES_BIODIESEL_PROD",
    "rfs_volumes": "EIA_SERIES_RFS_VOLUMES",
}


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get MotherDuck connection."""
    return duckdb.connect(f"md:{MOTHERDUCK_DB}")


def resolve_series_ids_from_env() -> Dict[str, str]:
    """Resolve logical series aliases to actual EIA series IDs via environment variables."""
    resolved: Dict[str, str] = {}
    for alias, env_var in SERIES_ENV_MAP.items():
        sid = os.getenv(env_var)
        if sid:
            resolved[alias] = sid
        else:
            logger.info(f"Env var {env_var} not set; skipping alias '{alias}'")
    return resolved


def get_last_loaded_date_for_series(
    con: duckdb.DuckDBPyConnection, series_id: str
) -> Optional[date]:
    """Get the last loaded date for a given EIA series."""
    query = f"""
    SELECT MAX(date) AS last_date
    FROM raw.eia_biofuels
    WHERE series_id = '{series_id}'
    """
    try:
        result = con.execute(query).fetchone()
        if result and result[0]:
            return result[0]
    except Exception as e:
        logger.warning(f"Could not determine last date for {series_id}: {e}")
    return None


def parse_eia_date(date_str: str) -> date:
    """Parse EIA series date strings into a Python date."""
    ds = date_str.strip()
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

    return pd.to_datetime(ds).date()


def fetch_eia_series(series_id: str, api_key: str) -> pd.DataFrame:
    """Fetch a single EIA series and return as DataFrame."""
    params = {
        "api_key": api_key,
        "series_id": series_id,
    }
    logger.info(f"Fetching EIA series {series_id}")
    resp = requests.get(EIA_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    series_list = data.get("series", [])
    if not series_list:
        logger.warning(f"No 'series' field in EIA response for {series_id}")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    series_obj = series_list[0]
    observations = series_obj.get("data", [])
    if not observations:
        logger.warning(f"No data points for series {series_id}")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    rows: List[Tuple[date, str, float]] = []
    for obs in observations:
        if not isinstance(obs, (list, tuple)) or len(obs) < 2:
            continue
        ds, val = obs[0], obs[1]
        try:
            dt = parse_eia_date(str(ds))
            fv = float(val)
            rows.append((dt, series_id, fv))
        except (TypeError, ValueError):
            continue

    if not rows:
        logger.warning(f"All observations for {series_id} were invalid")
        return pd.DataFrame(columns=["date", "series_id", "value"])

    df = pd.DataFrame(rows, columns=["date", "series_id", "value"])
    logger.info(f"Fetched {len(df)} rows for {series_id}")
    return df


def main():
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("EIA_API_KEY not set in environment")

    series_alias_to_id = resolve_series_ids_from_env()
    if not series_alias_to_id:
        raise RuntimeError(
            "No EIA series IDs configured. Set env vars: "
            "EIA_SERIES_RIN_D4, EIA_SERIES_RIN_D6, EIA_SERIES_BIODIESEL_PROD, EIA_SERIES_RFS_VOLUMES"
        )

    con = get_connection()
    all_frames: List[pd.DataFrame] = []

    for alias, series_id in series_alias_to_id.items():
        last_dt = get_last_loaded_date_for_series(con, series_id)
        df = fetch_eia_series(series_id, api_key)
        if df.empty:
            continue
        if last_dt is not None:
            df = df[df["date"] > last_dt]
            logger.info(
                f"After filtering, {len(df)} new rows for {series_id} (alias {alias})"
            )
        all_frames.append(df)

    if not all_frames:
        logger.info("No new EIA biofuels data to load.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["series_id", "date"]).drop_duplicates(
        ["series_id", "date"]
    )
    logger.info(f"Total new rows: {len(combined)}")

    # Save to Parquet
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = (
        PARQUET_DIR / f"eia_biofuels_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    )
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"Saved to {parquet_path}")

    # Upsert to MotherDuck
    con.execute(
        """
        INSERT OR REPLACE INTO raw.eia_biofuels (date, series_id, value)
        SELECT date, series_id, value FROM combined
    """
    )
    logger.info(f"✅ Upserted {len(combined)} rows to raw.eia_biofuels")


if __name__ == "__main__":
    main()
