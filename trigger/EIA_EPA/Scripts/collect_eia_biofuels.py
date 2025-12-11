#!/usr/bin/env python3
"""
EIA Biofuels Ingestion - MotherDuck/DuckDB (API v2)

Purpose: Collect EIA **biofuel feedstock** data critical for soybean oil PROCUREMENT decisions.

BIOFUEL FEEDSTOCKS (drives soybean oil demand):
  - Soybean oil inputs to biodiesel/renewable diesel production
  - Competing feedstocks (corn oil, yellow grease, tallow)
  - Biodiesel production capacity

NOTE:
- This job now **persists only biofuel/biodiesel-related series** into `raw.eia_biofuels`.
- Petroleum spot prices and refinery utilization endpoints are kept in helper
  functions but are **not** written to any raw table until a dedicated
  `raw.eia_energy`-style table is defined.

Output Tables:
  - raw.eia_biofuels (date, series_id, value)
  - data/raw/eia/eia_biofuels_YYYYMMDD.parquet
"""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
PARQUET_DIR = Path("/Volumes/Satechi Hub/CBI-V15/data/raw/eia")
EIA_V2_BASE = "https://api.eia.gov/v2"


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get MotherDuck connection."""
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={token}")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}")


def fetch_eia_v2(
    endpoint: str, params: dict[str, Any], api_key: str
) -> list[dict[str, Any]]:
    """Fetch data from EIA API v2."""
    url = f"{EIA_V2_BASE}{endpoint}"
    params["api_key"] = api_key
    logger.info(f"Fetching EIA v2: {endpoint}")

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    return data.get("response", {}).get("data", [])


def parse_period(period: str) -> date:
    """Parse EIA period string to date."""
    if "-" in period and len(period) == 10:  # YYYY-MM-DD
        return datetime.strptime(period, "%Y-%m-%d").date()
    if "-" in period and len(period) == 7:  # YYYY-MM
        return datetime.strptime(period, "%Y-%m").date()
    if len(period) == 4:  # YYYY
        return datetime.strptime(period, "%Y").date()
    return pd.to_datetime(period).date()


def collect_biofuel_feedstocks(api_key: str, start_date: str) -> pd.DataFrame:
    """Collect biofuel feedstock inputs (monthly)."""
    data = fetch_eia_v2(
        "/petroleum/pnp/feedbiofuel/data",
        {
            "frequency": "monthly",
            "data[0]": "value",
            "start": start_date[:7],
            "length": "1000",
        },
        api_key,
    )

    rows = []
    for row in data:
        if row.get("value") is None:
            continue
        period = row.get("period", "")
        rows.append(
            {
                "date": f"{period}-01" if len(period) == 7 else period,
                "series_id": row.get("series", ""),
                "value": float(row["value"]),
                "category": "biofuel_feedstock",
                "units": row.get("units", "MMLB"),
            }
        )

    logger.info(f"Collected {len(rows)} biofuel feedstock records")
    return pd.DataFrame(rows)


def collect_spot_prices(api_key: str, start_date: str) -> pd.DataFrame:
    """Collect petroleum spot prices (weekly)."""
    data = fetch_eia_v2(
        "/petroleum/pri/spt/data",
        {
            "frequency": "weekly",
            "data[0]": "value",
            "start": start_date,
            "length": "1000",
        },
        api_key,
    )

    rows = []
    for row in data:
        if row.get("value") is None:
            continue
        rows.append(
            {
                "date": row.get("period", ""),
                "series_id": row.get("series", ""),
                "value": float(row["value"]),
                "category": "spot_price",
                "units": row.get("units", "$/GAL"),
            }
        )

    logger.info(f"Collected {len(rows)} spot price records")
    return pd.DataFrame(rows)


def collect_refinery_utilization(api_key: str, start_date: str) -> pd.DataFrame:
    """Collect refinery utilization (weekly)."""
    data = fetch_eia_v2(
        "/petroleum/pnp/wiup/data",
        {
            "frequency": "weekly",
            "data[0]": "value",
            "start": start_date,
            "length": "500",
        },
        api_key,
    )

    rows = []
    for row in data:
        if row.get("value") is None:
            continue
        rows.append(
            {
                "date": row.get("period", ""),
                "series_id": row.get("series", ""),
                "value": float(row["value"]),
                "category": "refinery",
                "units": row.get("units", "%"),
            }
        )

    logger.info(f"Collected {len(rows)} refinery records")
    return pd.DataFrame(rows)


def main(days_back: int = 90) -> None:
    """Main entry point."""
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("EIA_API_KEY not set in environment")

    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    logger.info(f"Collecting EIA data from {start_date}")

    # Collect biofuel-related data only
    dfs = [collect_biofuel_feedstocks(api_key, start_date)]

    combined = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
    if combined.empty:
        logger.warning("No data collected")
        return

    combined = combined.drop_duplicates(["date", "series_id"])
    logger.info(f"Total records: {len(combined)}")

    # Save full collected data to Parquet (biofuels only for now)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = (
        PARQUET_DIR / f"eia_biofuels_{datetime.now().strftime('%Y%m%d')}.parquet"
    )
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"Saved to {parquet_path}")

    # Upsert to MotherDuck (biofuel feedstocks -> raw.eia_biofuels)
    try:
        con = get_connection()
        biofuels_df = combined[combined["category"] == "biofuel_feedstock"].copy()

        if biofuels_df.empty:
            logger.warning(
                "No biofuel feedstock records to upsert into raw.eia_biofuels"
            )
            con.close()
            return

        con.register("staging_df", biofuels_df)
        con.execute(
            """
            INSERT OR REPLACE INTO raw.eia_biofuels (date, series_id, value)
            SELECT
                CAST(date AS DATE),
                series_id,
                value
            FROM staging_df
        """
        )
        logger.info(f"✅ Upserted {len(biofuels_df)} rows to raw.eia_biofuels")
        con.close()
    except Exception as e:
        logger.error(f"MotherDuck upsert failed: {e}")
        logger.info("Data saved to Parquet only")


if __name__ == "__main__":
    main()
