#!/usr/bin/env python3
"""
NOAA Daily Weather Ingestion (US Midwest) - Mac-only, manual run

Bucket: weather_noaa

Source:
  - bigquery-public-data.ghcn_d (NOAA GHCN-D daily)

Target:
  - raw.weather_noaa (date, station_id, region, metric, value)
    with monthly partitioning on date and clustering by region.

Pattern:
  - Read from public NOAA GHCN-D tables (station-level).
  - Restrict to key Midwest states (IL, IA, IN, NE, OH) for now.
  - Keep station-level granularity:
      station_id  = NOAA station ID (GHCN)
      region      = state code (e.g., 'IL')
      metric      = one of: 'tavg_c', 'tmax_c', 'tmin_c', 'prcp_mm'
      value       = numeric value in canonical units.
  - Write to raw_staging.weather_noaa_<run_id>, then MERGE into raw.weather_noaa
    on (station_id, date, metric).
"""

import sys
from pathlib import Path
from datetime import datetime, date
from typing import List

import logging
import pandas as pd
from google.cloud import bigquery

# Project root for utils import
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cbi_utils.bigquery_client import (  # noqa: E402
    get_client,
    load_dataframe_to_table,
    merge_staging_to_target,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
RAW_TABLE = f"{PROJECT_ID}.raw.weather_noaa"
STAGING_DATASET = f"{PROJECT_ID}.raw_staging"

# Midwest region states (can be extended later)
MIDWEST_STATES: List[str] = ["IL", "IA", "IN", "NE", "OH"]


def build_query(start_year: int, end_year: int) -> str:
    """
    Build a BigQuery SQL statement that reads NOAA GHCN-D daily
    data for the specified year range and Midwest states.
    """
    state_list = ",".join(f"'{s}'" for s in MIDWEST_STATES)
    return f"""
    SELECT
      d.date,
      d.id AS station_id,
      st.code AS region,
      CASE
        WHEN d.element = 'TAVG' THEN 'tavg_c'
        WHEN d.element = 'TMAX' THEN 'tmax_c'
        WHEN d.element = 'TMIN' THEN 'tmin_c'
        WHEN d.element = 'PRCP' THEN 'prcp_mm'
        ELSE NULL
      END AS metric,
      CASE
        WHEN d.element IN ('TAVG','TMAX','TMIN') THEN d.value / 10.0    -- tenths of °C -> °C
        WHEN d.element = 'PRCP' THEN d.value / 10.0                     -- tenths of mm -> mm
        ELSE NULL
      END AS value
    FROM `bigquery-public-data.ghcn_d.ghcnd_*` AS d
    JOIN `bigquery-public-data.ghcn_d.ghcnd_stations` AS s
      ON d.id = s.id
    JOIN `bigquery-public-data.ghcn_d.ghcnd_states` AS st
      ON s.state = st.code
    WHERE
      _TABLE_SUFFIX BETWEEN '{start_year}' AND '{end_year}'
      AND d.element IN ('TAVG','TMAX','TMIN','PRCP')
      AND st.code IN ({state_list})
      AND d.date >= DATE('{start_year}-01-01')
    """


def main(start_year: int = 2005, end_year: int = datetime.utcnow().year) -> None:
    client = get_client(project_id=PROJECT_ID)

    logger.info(
        f"[weather_noaa] Ingesting NOAA GHCN-D for Midwest states "
        f"{MIDWEST_STATES} from {start_year} to {end_year}"
    )

    sql = build_query(start_year, end_year)
    job_config = bigquery.QueryJobConfig()
    df = client.query(sql, job_config=job_config, location="US").to_dataframe()

    if df.empty:
        logger.warning("[weather_noaa] Query returned no rows.")
        return

    # Drop rows with unknown metrics or values
    df = df.dropna(subset=["metric", "value"])

    # Enforce dtypes
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["station_id"] = df["station_id"].astype(str)
    df["region"] = df["region"].astype(str)
    df["metric"] = df["metric"].astype(str)
    df["value"] = df["value"].astype(float)

    logger.info(f"[weather_noaa] Retrieved {len(df):,} rows from public NOAA dataset")

    # Ensure staging dataset exists
    try:
        client.get_dataset(STAGING_DATASET)
    except Exception:
        client.create_dataset(bigquery.Dataset(STAGING_DATASET))

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    staging_table = f"{STAGING_DATASET}.weather_noaa_{run_id}"

    if not load_dataframe_to_table(df, staging_table, PROJECT_ID, "WRITE_TRUNCATE"):
        logger.error(f"[weather_noaa] ❌ Failed to load into staging table {staging_table}")
        return

    # MERGE into canonical raw.weather_noaa
    if merge_staging_to_target(
        staging_table=staging_table,
        target_table=RAW_TABLE,
        key_columns=["station_id", "date", "metric"],
        all_columns=["date", "station_id", "region", "metric", "value"],
        project_id=PROJECT_ID,
    ):
        logger.info(f"[weather_noaa] ✅ MERGE from {staging_table} into {RAW_TABLE} complete")
    else:
        logger.error(f"[weather_noaa] ❌ MERGE from {staging_table} into {RAW_TABLE} failed")


if __name__ == "__main__":
    main()
