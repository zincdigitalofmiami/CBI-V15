#!/usr/bin/env python3
"""
ScrapeCreators News Ingestion - Canonical Bucketed Pipeline (MotherDuck Edition)

Role:
- Pull news items from ScrapeCreators (via per-bucket collectors).
- Segment them into the canonical news bucket schema.
- Save to External Drive (Parquet) -> Load to MotherDuck (raw.scrapecreators_news_buckets).

Contract:
- Source: ScrapeCreators API (via bucket modules)
- Sink 1: /Volumes/Satechi Hub/CBI-V15/data/raw/scrape_creator/{bucket}/{date}.parquet
- Sink 2: MotherDuck raw.scrapecreators_news_buckets

POLICY: REAL DATA ONLY - NO MOCKS EVER.
"""

import os
import sys
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import bucket collectors
from src.ingestion.scrape_creator.buckets import (
    collect_biofuel_policy,
    collect_china_demand,
    collect_tariffs_trade_policy,
    collect_trump_truth_social,
)

# Configuration
DRIVE_ROOT = Path("/Volumes/Satechi Hub/CBI-V15/data/raw/scrape_creator")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")


def fetch_all_buckets() -> List[Dict[str, Any]]:
    """Fetch items from all ScrapeCreators buckets."""
    all_items = []

    bucket_modules = [
        ("biofuel_policy", collect_biofuel_policy),
        ("china_demand", collect_china_demand),
        ("tariffs_trade_policy", collect_tariffs_trade_policy),
        ("trump_truth_social", collect_trump_truth_social),
    ]

    for bucket_name, module in bucket_modules:
        try:
            print(f"Fetching bucket: {bucket_name}...")
            if hasattr(module, "fetch_bucket_items"):
                items = module.fetch_bucket_items()
                for it in items:
                    it["bucket_name"] = bucket_name
                    all_items.append(it)
            else:
                print(f"Warning: {bucket_name} missing fetch_bucket_items")
        except Exception as e:
            print(f"Error fetching {bucket_name}: {e}")

    return all_items


def save_to_drive(df: pd.DataFrame, run_id: str):
    """Save raw data to External Drive organized by bucket."""
    if df.empty:
        return

    for bucket_name in df["bucket_name"].unique():
        bucket_df = df[df["bucket_name"] == bucket_name]

        output_dir = DRIVE_ROOT / bucket_name
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{datetime.now().strftime('%Y%m%d')}_{run_id}.parquet"
        output_path = output_dir / filename

        print(f"Saving {len(bucket_df)} rows to {output_path}")
        bucket_df.to_parquet(output_path, index=False)


def load_to_motherduck(df: pd.DataFrame):
    """Load data into MotherDuck raw schema."""
    if df.empty:
        return

    print("Connecting to MotherDuck...")
    con = duckdb.connect(f"md:{MOTHERDUCK_DB}")

    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw.scrapecreators_news_buckets (
            article_id TEXT,
            bucket_name TEXT,
            headline TEXT,
            content TEXT,
            source TEXT,
            published_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (article_id)
        )
    """
    )

    print(f"Loading {len(df)} rows to MotherDuck...")
    con.register("df_view", df)
    con.execute(
        """
        INSERT INTO raw.scrapecreators_news_buckets 
        SELECT 
            article_id, bucket_name, headline, content, source, 
            CAST(published_at AS TIMESTAMP), current_timestamp 
        FROM df_view
        ON CONFLICT DO NOTHING
    """
    )
    print("Load complete.")


def main():
    print("Starting ScrapeCreator Ingestion...")
    run_id = datetime.now().strftime("%H%M%S")

    items = fetch_all_buckets()
    if not items:
        print("No items fetched.")
        return

    df = pd.DataFrame(items)

    save_to_drive(df, run_id)
    load_to_motherduck(df)

    print("Ingestion Pipeline Complete.")


if __name__ == "__main__":
    main()
