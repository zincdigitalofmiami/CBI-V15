#!/usr/bin/env python3
"""
FRED Priority Series Collector
Collects 30+ key price/rate series with COMPLETE historical data

Priority series:
- 15 daily series (WTI, FX rates, Fed Funds, VIX)
- 5 weekly series (Diesel, Gasoline, NFCI, STLFSI)
- 10 monthly series (Soybeans, Corn, Copper, CPI, Unemployment)

Total: 30 series × 10,000 avg observations = 300,000 data points
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import duckdb
import pandas as pd
import requests
import yaml
from pathlib import Path

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
FRED_API_KEY = os.getenv("FRED_API_KEY")

FRED_OBS_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"

# Load config
CONFIG_PATH = Path("/Volumes/Satechi Hub/CBI-V15/config/fred_price_series.yaml")


def load_priority_series() -> Dict[str, List[str]]:
    """Load priority series from config."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return {
        "daily": config.get("priority_daily", []),
        "weekly": config.get("priority_weekly", []),
        "monthly": config.get("priority_monthly", []),
    }


def fetch_series_observations(series_id: str) -> List[Dict[str, Any]]:
    """Fetch ALL observations for a FRED series."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": 100000,  # Max observations per request
    }

    try:
        resp = requests.get(FRED_OBS_ENDPOINT, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])

        # Filter out missing values (represented as ".")
        valid_obs = [o for o in observations if o["value"] != "."]

        return valid_obs

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []


def parse_observations_to_rows(
    series_id: str, observations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert FRED observations to database rows."""
    rows = []

    for obs in observations:
        date_str = obs.get("date")
        value_str = obs.get("value")

        try:
            value = float(value_str)
            date = pd.to_datetime(date_str).date()

            rows.append(
                {
                    "series_id": series_id,
                    "date": date,
                    "value": value,
                    "source": "fred",
                    "ingested_at": datetime.now(),
                }
            )
        except (ValueError, TypeError):
            continue

    return rows


def load_to_motherduck(rows: List[Dict[str, Any]]) -> int:
    """Load rows to MotherDuck."""
    if not rows:
        return 0

    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    df = pd.DataFrame(rows)
    con.register("temp_data", df)

    # Upsert to handle updates
    con.execute(
        """
        INSERT INTO raw.fred_economic
        SELECT * FROM temp_data
        ON CONFLICT (series_id, date) DO UPDATE SET
            value = EXCLUDED.value,
            ingested_at = EXCLUDED.ingested_at
    """
    )

    count = con.execute("SELECT COUNT(*) FROM raw.fred_economic").fetchone()[0]
    con.close()

    return count


def main():
    """Main collection routine."""
    print("=" * 80)
    print("FRED PRIORITY SERIES COLLECTOR")
    print("=" * 80)

    if not FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY required - get free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    # Load priority series
    priority = load_priority_series()

    all_rows = []
    series_stats = {}

    # Collect daily series
    print("\n[1/3] DAILY SERIES (15 series)")
    print("=" * 80)
    for series_id in priority["daily"]:
        print(f"\n{series_id}...")
        observations = fetch_series_observations(series_id)

        if observations:
            rows = parse_observations_to_rows(series_id, observations)
            all_rows.extend(rows)
            series_stats[series_id] = len(rows)

            first = observations[0]
            last = observations[-1]
            print(f"  ✅ {len(rows):,} observations")
            print(f"  Range: {first['date']} to {last['date']}")
            print(f"  Latest: {last['value']}")

    # Collect weekly series
    print("\n[2/3] WEEKLY SERIES (5 series)")
    print("=" * 80)
    for series_id in priority["weekly"]:
        print(f"\n{series_id}...")
        observations = fetch_series_observations(series_id)

        if observations:
            rows = parse_observations_to_rows(series_id, observations)
            all_rows.extend(rows)
            series_stats[series_id] = len(rows)

            first = observations[0]
            last = observations[-1]
            print(f"  ✅ {len(rows):,} observations")
            print(f"  Range: {first['date']} to {last['date']}")

    # Collect monthly series
    print("\n[3/3] MONTHLY SERIES (10 series)")
    print("=" * 80)
    for series_id in priority["monthly"]:
        print(f"\n{series_id}...")
        observations = fetch_series_observations(series_id)

        if observations:
            rows = parse_observations_to_rows(series_id, observations)
            all_rows.extend(rows)
            series_stats[series_id] = len(rows)

            first = observations[0]
            last = observations[-1]
            print(f"  ✅ {len(rows):,} observations")
            print(f"  Range: {first['date']} to {last['date']}")

    # Load to MotherDuck
    print("\n" + "=" * 80)
    print("LOADING TO MOTHERDUCK")
    print("=" * 80)
    print(f"Total observations: {len(all_rows):,}")
    print(f"Unique series: {len(series_stats)}")

    if all_rows:
        total_count = load_to_motherduck(all_rows)
        print(f"\n✅ raw.fred_economic: {total_count:,} total rows")

        # Show summary by frequency
        con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

        summary = con.execute(
            """
            SELECT 
                series_id,
                COUNT(*) as obs_count,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM raw.fred_economic
            WHERE series_id IN (
                'DCOILWTICO', 'DEXBZUS', 'DEXCHUS', 'VIXCLS', 
                'DFF', 'DGS10', 'T10Y2Y', 'NFCI', 'STLFSI4'
            )
            GROUP BY series_id
            ORDER BY obs_count DESC
        """
        ).fetchall()

        print("\nKey Series Summary:")
        for row in summary:
            print(f"  {row[0]:15} {row[1]:6,} obs  {row[2]} to {row[3]}")

        con.close()
    else:
        print("\n⚠️  No observations collected")


if __name__ == "__main__":
    main()
