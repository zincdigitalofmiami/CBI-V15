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

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
FRED_API_KEY: str | None = None

FRED_OBS_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_ENDPOINT = "https://api.stlouisfed.org/fred/series"

# Load config
ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "config" / "fred_price_series.yaml"


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def _load_local_env() -> None:
    _load_dotenv_file(ROOT_DIR / ".env")
    _load_dotenv_file(ROOT_DIR / ".env.local")


def _iter_motherduck_tokens():
    candidates = [
        ("MOTHERDUCK_TOKEN", os.getenv("MOTHERDUCK_TOKEN")),
        ("motherduck_storage_MOTHERDUCK_TOKEN", os.getenv("motherduck_storage_MOTHERDUCK_TOKEN")),
        ("MOTHERDUCK_READ_SCALING_TOKEN", os.getenv("MOTHERDUCK_READ_SCALING_TOKEN")),
        (
            "motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN",
            os.getenv("motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN"),
        ),
    ]
    for _, value in candidates:
        if not value:
            continue
        token = value.strip().strip('"').strip("'")
        if token.count(".") != 2:
            continue
        yield token


def connect_motherduck() -> duckdb.DuckDBPyConnection:
    _load_local_env()
    db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    last_error: Exception | None = None
    for token in _iter_motherduck_tokens():
        try:
            con = duckdb.connect(f"md:{db_name}?motherduck_token={token}")
            con.execute("SELECT 1").fetchone()
            return con
        except Exception as e:
            last_error = e
    raise ValueError(f"MotherDuck token required (set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN): {last_error}")


# Load local env early so FRED_API_KEY is available when running as a script.
_load_local_env()
FRED_API_KEY = os.getenv("FRED_API_KEY")


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

def fetch_series_metadata(series_id: str) -> Dict[str, Any] | None:
    """Fetch series metadata from FRED."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    try:
        resp = requests.get(FRED_SERIES_ENDPOINT, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        seriess = payload.get("seriess") or payload.get("series") or []
        if not seriess:
            return None
        s = seriess[0]
        return {
            "series_id": series_id,
            "title": s.get("title"),
            "category": None,
            "frequency": s.get("frequency"),
            "units": s.get("units"),
            "seasonal_adjustment": s.get("seasonal_adjustment"),
            "observation_start": pd.to_datetime(s.get("observation_start")).date() if s.get("observation_start") else None,
            "observation_end": pd.to_datetime(s.get("observation_end")).date() if s.get("observation_end") else None,
            "last_updated": pd.to_datetime(s.get("last_updated")) if s.get("last_updated") else None,
            "popularity": int(s.get("popularity")) if s.get("popularity") not in (None, "") else None,
            "notes": s.get("notes"),
            "discovered_at": datetime.now(),
            "source": "fred",
            "ingested_at": datetime.now(),
        }
    except Exception:
        return None


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
    con = connect_motherduck()

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

def load_metadata_to_motherduck(metadata_rows: List[Dict[str, Any]]) -> int:
    if not metadata_rows:
        return 0
    con = connect_motherduck()
    df = pd.DataFrame(metadata_rows)
    con.register("temp_meta", df)
    # Robust upsert: delete+insert (MotherDuck constraints are not always visible to ON CONFLICT)
    con.execute(
        """
        DELETE FROM raw.fred_series_metadata
        WHERE series_id IN (SELECT series_id FROM temp_meta)
        """
    )
    con.execute(
        """
        INSERT INTO raw.fred_series_metadata (
          series_id, title, category, frequency, units, seasonal_adjustment,
          observation_start, observation_end, last_updated, popularity, notes,
          discovered_at, source, ingested_at
        )
        SELECT
          series_id, title, category, frequency, units, seasonal_adjustment,
          observation_start, observation_end, last_updated, popularity, notes,
          discovered_at, source, ingested_at
        FROM temp_meta
        """
    )
    cnt = con.execute("SELECT COUNT(*) FROM raw.fred_series_metadata").fetchone()[0]
    con.close()
    return cnt

def main():
    """Main collection routine."""
    print("=" * 80)
    print("FRED PRIORITY SERIES COLLECTOR")
    print("=" * 80)

    if not FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY required - get free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only refresh raw.fred_series_metadata; skip downloading observations",
    )
    args = parser.parse_args()

    # Load priority series
    priority = load_priority_series()

    all_rows = []
    series_stats = {}
    meta_rows: List[Dict[str, Any]] = []

    # Collect daily series
    print("\n[1/3] DAILY SERIES (15 series)")
    print("=" * 80)
    for series_id in priority["daily"]:
        print(f"\n{series_id}...")
        meta = fetch_series_metadata(series_id)
        if meta:
            meta_rows.append(meta)
        observations = [] if args.metadata_only else fetch_series_observations(series_id)

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
        meta = fetch_series_metadata(series_id)
        if meta:
            meta_rows.append(meta)
        observations = [] if args.metadata_only else fetch_series_observations(series_id)

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
        meta = fetch_series_metadata(series_id)
        if meta:
            meta_rows.append(meta)
        observations = [] if args.metadata_only else fetch_series_observations(series_id)

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
    else:
        print("\n⚠️  No observations collected")

    if meta_rows:
        meta_count = load_metadata_to_motherduck(meta_rows)
        print(f"✅ raw.fred_series_metadata: {meta_count:,} total rows")

    # Show summary by frequency
    con = connect_motherduck()

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


if __name__ == "__main__":
    main()
