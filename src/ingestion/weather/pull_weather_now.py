#!/usr/bin/env python3
"""
Weather Data Puller - Uses weather.gov API (FREE, NO TOKEN NEEDED)
API: https://www.weather.gov/documentation/services-web-api

This API is FREE and requires NO TOKEN - just a User-Agent header!
Covers US weather stations with current observations.

For Brazil/Argentina, we'll use NOAA GHCN Daily API (requires free token).
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import duckdb
import pandas as pd
import requests

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# weather.gov API (FREE, no token!)
WEATHER_GOV_API = "https://api.weather.gov"

# US Agricultural weather stations (Corn Belt)
US_STATIONS = {
    "KDSM": {
        "name": "Des Moines, IA",
        "lat": 41.5339,
        "lon": -93.6631,
        "region": "Iowa",
    },
    "KORD": {
        "name": "Chicago O'Hare, IL",
        "lat": 41.9742,
        "lon": -87.9073,
        "region": "Illinois",
    },
    "KMSP": {
        "name": "Minneapolis, MN",
        "lat": 44.8848,
        "lon": -93.2223,
        "region": "Minnesota",
    },
    "KOMA": {
        "name": "Omaha, NE",
        "lat": 41.3032,
        "lon": -95.8942,
        "region": "Nebraska",
    },
}


def fetch_station_observation(station_id: str) -> Dict[str, Any]:
    """
    Fetch latest observation from weather.gov API.
    NO TOKEN NEEDED - just User-Agent header!
    """
    headers = {
        "User-Agent": "(CBI-V15 Weather Collector, zincmiami@gmail.com)",
        "Accept": "application/geo+json",
    }

    url = f"{WEATHER_GOV_API}/stations/{station_id}/observations/latest"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        props = data.get("properties", {})
        return props

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {}


def parse_observation(
    obs: Dict[str, Any], station_id: str, station_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Parse weather.gov observation to database row."""
    if not obs:
        return None

    # Extract timestamp
    timestamp = obs.get("timestamp")
    if not timestamp:
        return None

    date = pd.to_datetime(timestamp).date()

    # Extract weather values
    temp = obs.get("temperature", {})
    precip = obs.get("precipitationLastHour", {})

    temp_c = temp.get("value") if temp else None
    precip_mm = precip.get("value") if precip else None

    return {
        "station_id": station_id,
        "date": date,
        "tavg_c": temp_c,
        "tmin_c": None,  # Not in latest obs
        "tmax_c": None,  # Not in latest obs
        "prcp_mm": precip_mm,
        "snow_mm": None,
        "region": station_info["region"],
        "country": "US",
        "source": "weather_gov_api",
        "ingested_at": datetime.now(),
    }


def load_to_motherduck(rows: List[Dict[str, Any]]) -> int:
    """Load to MotherDuck."""
    if not rows:
        return 0

    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    df = pd.DataFrame(rows)
    con.register("temp_data", df)

    con.execute(
        """
        INSERT INTO raw.weather_noaa
        SELECT * FROM temp_data
        ON CONFLICT (station_id, date) DO UPDATE SET
            tavg_c = EXCLUDED.tavg_c,
            prcp_mm = EXCLUDED.prcp_mm,
            ingested_at = EXCLUDED.ingested_at
    """
    )

    count = con.execute("SELECT COUNT(*) FROM raw.weather_noaa").fetchone()[0]
    con.close()

    return count


def main():
    """Pull current weather observations."""
    print("=" * 80)
    print("WEATHER.GOV API - CURRENT OBSERVATIONS (FREE, NO TOKEN)")
    print("=" * 80)

    rows = []

    for station_id, info in US_STATIONS.items():
        print(f"\n{info['name']}...")
        obs = fetch_station_observation(station_id)

        if obs:
            row = parse_observation(obs, station_id, info)
            if row:
                rows.append(row)
                print(
                    f"  ✅ Temp: {row['tavg_c']:.1f}°C  Precip: {row['prcp_mm']:.1f}mm"
                    if row["tavg_c"] and row["prcp_mm"]
                    else "  ✅ Data collected"
                )

    if rows:
        total_count = load_to_motherduck(rows)
        print(f"\n✅ Loaded {len(rows)} observations")
        print(f"Total in database: {total_count:,} rows")
    else:
        print("\n❌ No observations collected")


if __name__ == "__main__":
    main()
