#!/usr/bin/env python3
"""
FRED Historical Release Backfill
Uses FRED API release/observations endpoint to get COMPLETE historical data
for all Fed monetary policy, financial conditions, and macro indicators

This is GOLD - gets us entire history of all series in one go!

API Docs: https://fred.stlouisfed.org/docs/api/fred/v2/release_observations.html
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import duckdb
import pandas as pd
import requests

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# FRED API endpoints
FRED_BASE = "https://api.stlouisfed.org/fred"
RELEASE_OBS_ENDPOINT = f"{FRED_BASE}/release/observations"
TAGS_ENDPOINT = f"{FRED_BASE}/tags"
RELATED_TAGS_ENDPOINT = f"{FRED_BASE}/related_tags"
TAGS_SERIES_ENDPOINT = f"{FRED_BASE}/tags/series"

# Key FRED Releases for Big 8 Buckets
FRED_RELEASES = {
    "fed_monetary": {
        "release_id": 62,  # H.15 Selected Interest Rates
        "description": "Fed Funds, Treasury Yields, SOFR, IORB",
        "bucket": "fed",
    },
    "financial_conditions": {
        "release_id": 469,  # Financial Stress Indices
        "description": "NFCI, STLFSI, Financial Conditions",
        "bucket": "volatility",
    },
    "z1_financial_accounts": {
        "release_id": 52,  # Z.1 Financial Accounts
        "description": "Flow of Funds, Credit Markets",
        "bucket": "fed",
    },
}

# Tags for targeted series discovery
FRED_TAGS = {
    "fed": ["monetary policy", "interest rate", "federal funds", "yield curve"],
    "volatility": ["volatility", "vix", "financial stress", "credit spread"],
    "fx": ["exchange rate", "dollar index", "currency"],
    "energy": ["oil", "petroleum", "energy prices"],
}


def fetch_release_observations(
    release_id: int, limit: int = 500000
) -> List[Dict[str, Any]]:
    """
    Fetch ALL observations for a FRED release.
    Handles pagination automatically using next_cursor.

    This is the GOLD endpoint - gets entire history in one call!
    """
    params = {
        "release_id": release_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
    }

    all_series = []
    next_cursor = None
    page = 1

    while True:
        if next_cursor:
            params["next_cursor"] = next_cursor

        print(f"  Page {page}: Fetching...")

        try:
            resp = requests.get(RELEASE_OBS_ENDPOINT, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # Extract series
            series_list = data.get("series", [])
            all_series.extend(series_list)

            print(f"  Page {page}: Got {len(series_list)} series")

            # Check for more pages
            has_more = data.get("has_more", False)
            if not has_more:
                break

            next_cursor = data.get("next_cursor")
            if not next_cursor:
                break

            page += 1

        except Exception as e:
            print(f"  ❌ Error on page {page}: {e}")
            break

    return all_series


def parse_series_to_rows(
    series_data: List[Dict[str, Any]], bucket: str
) -> List[Dict[str, Any]]:
    """Convert FRED series data to database rows."""
    rows = []

    for series in series_data:
        series_id = series.get("series_id")
        observations = series.get("observations", [])

        for obs in observations:
            date_str = obs.get("date")
            value_str = obs.get("value")

            # Skip missing values (represented as ".")
            if value_str == ".":
                continue

            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue

            rows.append(
                {
                    "series_id": series_id,
                    "date": pd.to_datetime(date_str).date(),
                    "value": value,
                    "source": "fred",
                    "ingested_at": datetime.now(),
                }
            )

    return rows


def fetch_series_by_tags(tags: List[str], limit: int = 1000) -> List[str]:
    """
    Discover series IDs by tags.
    Example: tags=['monetary policy', 'interest rate'] finds all Fed policy series.
    """
    params = {
        "tag_names": ";".join(tags),
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
    }

    try:
        resp = requests.get(TAGS_SERIES_ENDPOINT, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        series_list = data.get("seriess", [])
        series_ids = [s.get("id") for s in series_list]

        return series_ids

    except Exception as e:
        print(f"  ❌ Error fetching tags: {e}")
        return []


def load_to_motherduck(rows: List[Dict[str, Any]]) -> int:
    """Load rows to MotherDuck."""
    if not rows:
        return 0

    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    df = pd.DataFrame(rows)
    con.register("temp_data", df)

    # Insert with ON CONFLICT to avoid duplicates
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
    """Main backfill routine."""
    print("=" * 80)
    print("FRED HISTORICAL BACKFILL (Complete History)")
    print("=" * 80)

    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY required")

    all_rows = []

    # 1. Fetch by Releases (Gets EVERYTHING in one call)
    print("\n[1/2] Fetching FRED Releases (Complete History)...")
    for release_name, config in FRED_RELEASES.items():
        release_id = config["release_id"]
        description = config["description"]
        bucket = config["bucket"]

        print(f"\n{release_name.upper()}")
        print(f"  Release ID: {release_id}")
        print(f"  Description: {description}")

        series_data = fetch_release_observations(release_id)
        rows = parse_series_to_rows(series_data, bucket)
        all_rows.extend(rows)

        print(f"  ✅ Collected {len(rows):,} observations")

    # 2. Fetch by Tags (Discover additional series)
    print("\n[2/2] Discovering Series by Tags...")
    for bucket, tags in FRED_TAGS.items():
        print(f"\n{bucket.upper()}: {', '.join(tags)}")
        series_ids = fetch_series_by_tags(tags)
        print(f"  ✅ Found {len(series_ids)} series")

    # Load to MotherDuck
    print("\n" + "=" * 80)
    print("LOADING TO MOTHERDUCK")
    print("=" * 80)
    print(f"Total observations collected: {len(all_rows):,}")

    if all_rows:
        total_count = load_to_motherduck(all_rows)
        print(f"\n✅ raw.fred_economic: {total_count:,} total rows")

        # Show date range
        con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
        date_range = con.execute(
            """
            SELECT 
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT series_id) as unique_series
            FROM raw.fred_economic
        """
        ).fetchone()
        con.close()

        print(f"\nData Range: {date_range[0]} to {date_range[1]}")
        print(f"Unique Series: {date_range[2]}")
    else:
        print("\n⚠️  No observations collected")


if __name__ == "__main__":
    main()
