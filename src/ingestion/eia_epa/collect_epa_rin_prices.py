#!/usr/bin/env python3
"""
EPA RIN Prices Ingestion
Collects weekly RIN prices (D3, D4, D5, D6) for biofuel tracking.
Target: raw.epa_rin_prices

DATA SOURCES (in priority order):
1. OPIS_API_KEY env var - OPIS daily RIN prices (paid, ~$2000/year)
2. EPA EMTS - EPA Moderated Transaction System (requires EPA registration)

RIN Types:
- D3: Cellulosic biofuel (highest value)
- D4: Biomass-based diesel (biodiesel, renewable diesel)
- D5: Advanced biofuel (sugarcane ethanol)
- D6: Renewable fuel (corn ethanol, lowest value)

CRITICAL for ZL forecasting: D4 RINs drive soybean oil demand for biodiesel.
"""
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
ROOT_DIR = Path(__file__).resolve().parents[3]
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# API keys (set in environment for production)
OPIS_API_KEY = os.getenv("OPIS_API_KEY")  # OPIS RIN prices (paid)


def get_connection():
    """Get MotherDuck connection (CLOUD ONLY - no local fallback)"""
    if not MOTHERDUCK_TOKEN:
        raise RuntimeError("MOTHERDUCK_TOKEN not set - cannot proceed (CLOUD ONLY)")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")


def fetch_opis_rin_prices():
    """
    Fetch RIN prices from OPIS API (paid subscription required).
    Returns: DataFrame with columns: date, rin_type, price or None if unavailable
    """
    if not OPIS_API_KEY:
        logger.info("OPIS_API_KEY not set - OPIS data unavailable")
        return None

    logger.info("Fetching RIN prices from OPIS API...")

    try:
        # OPIS API endpoint (structure varies by subscription)
        # This is a placeholder - actual endpoint depends on OPIS contract
        headers = {"Authorization": f"Bearer {OPIS_API_KEY}"}
        response = requests.get(
            "https://api.opisnet.com/rin-prices/daily", headers=headers, timeout=30
        )
        response.raise_for_status()

        data = response.json()
        records = []

        for item in data.get("prices", []):
            records.append(
                {
                    "date": pd.to_datetime(item["date"]).date(),
                    "rin_type": item["rin_type"],
                    "price": float(item["price"]),
                }
            )

        df = pd.DataFrame(records)
        logger.info(f"✅ Fetched {len(df)} records from OPIS")
        return df

    except requests.exceptions.RequestException as e:
        logger.warning(f"OPIS API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing OPIS data: {e}")
        return None


def get_rin_prices():
    """
    Get RIN prices from best available source.
    Priority: OPIS API > (unimplemented EPA EMTS)

    NOTE: This repo prohibits synthetic/placeholder data. If no real source is
    configured, this returns an empty DataFrame and does not write to the DB.
    """
    # Try OPIS first (paid, real-time)
    df = fetch_opis_rin_prices()
    if df is not None and not df.empty:
        return df

    logger.info("No real RIN source configured (OPIS_API_KEY missing); skipping")
    return pd.DataFrame(columns=["date", "rin_type", "price"])


def main():
    """Main ingestion function"""
    logger.info("=" * 60)
    logger.info("EPA RIN PRICES INGESTION")
    logger.info("=" * 60)

    # Get RIN prices from best available source
    df = get_rin_prices()
    logger.info(f"Collected {len(df)} RIN price records")

    if df.empty:
        logger.warning("No real data to ingest (skipping write)")
        return

    # Connect to database
    conn = get_connection()

    # Load to MotherDuck
    logger.info("Loading to raw.epa_rin_prices...")

    # Register DataFrame
    conn.register("staging_rin", df)

    # Delete existing data for these dates (idempotent upsert)
    min_date = df["date"].min()
    max_date = df["date"].max()
    conn.execute(
        f"""
        DELETE FROM raw.epa_rin_prices
        WHERE date >= '{min_date}' AND date <= '{max_date}'
    """
    )

    # Insert new data
    conn.execute(
        """
        INSERT INTO raw.epa_rin_prices (date, rin_type, price)
        SELECT date, rin_type, price
        FROM staging_rin
    """
    )

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM raw.epa_rin_prices").fetchone()[0]
    logger.info(f"✅ Loaded {len(df)} rows to raw.epa_rin_prices (total: {count})")

    conn.close()


if __name__ == "__main__":
    main()
