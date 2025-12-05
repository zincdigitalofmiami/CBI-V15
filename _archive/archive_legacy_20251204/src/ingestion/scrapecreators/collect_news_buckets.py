#!/usr/bin/env python3
"""
ScrapeCreators News Ingestion - Canonical Bucketed Pipeline

Role:
- Pull news items from ScrapeCreators (via per-bucket collectors).
- Segment them into the canonical news bucket schema:
  raw.scrapecreators_news_buckets (theme_primary, is_trump_related, policy_axis, horizon, zl_sentiment, impact_magnitude, etc.).
- Use raw_staging + MERGE to keep ingestion idempotent (no duplicates).

IMPORTANT:
- This script does NOT fabricate data. It assumes the per-bucket fetch_* functions
  are wired to the real ScrapeCreators API and returns real items.
- Until those are implemented, running this will raise NotImplementedError upstream.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from google.cloud import bigquery

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.scrapecreators.buckets import (  # noqa: E402
    collect_biofuel_policy,
    collect_china_demand,
    collect_tariffs_trade_policy,
    collect_trump_truth_social,
)
from src.ingestion.scrapecreators.sentiment_calculator import (  # noqa: E402
    calculate_sentiment_finbert,
)

PROJECT_ID = os.getenv("GCP_PROJECT", "cbi-v15")
RAW_TABLE = f"{PROJECT_ID}.raw.scrapecreators_news_buckets"
STAGING_DATASET = f"{PROJECT_ID}.raw_staging"


def ensure_staging_dataset(client: bigquery.Client) -> None:
    """Ensure raw_staging dataset exists."""
    ds_ref = bigquery.Dataset(STAGING_DATASET)
    try:
        client.get_dataset(ds_ref)
    except Exception:
        client.create_dataset(ds_ref)


def fetch_all_buckets() -> List[Dict[str, Any]]:
    """
    Fetch items from all ScrapeCreators buckets.
    Each bucket module defines fetch_bucket_items() which must hit the real API.
    """
    all_items: List[Dict[str, Any]] = []

    bucket_modules = [
        ("biofuel_policy", collect_biofuel_policy),
        ("china_demand", collect_china_demand),
        ("tariffs_trade_policy", collect_tariffs_trade_policy),
        ("trump_truth_social", collect_trump_truth_social),
    ]

    for bucket_name, module in bucket_modules:
        fetch_fn = getattr(module, "fetch_bucket_items", None)
        if fetch_fn is None:
            raise RuntimeError(f"Bucket module {module.__name__} missing fetch_bucket_items")

        items = fetch_fn()
        if not isinstance(items, list):
            raise RuntimeError(f"{module.__name__}.fetch_bucket_items must return a list of dicts")

        for it in items:
            it["bucket_name"] = bucket_name
            all_items.append(it)

    return all_items


def map_theme_and_policy(bucket_name: str, text: str) -> Dict[str, Any]:
    """
    Map bucket_name + text to theme_primary, is_trump_related, policy_axis, horizon.
    This is rule-based; improve as needed.
    """
    bn = bucket_name.lower()
    theme_primary = None
    is_trump_related = False
    policy_axis = None
    horizon = "TACTICAL"

    if "biofuel" in bn:
        theme_primary = "DEMAND_BIOFUELS"
        policy_axis = "BIOFUELS_RFS"
        horizon = "STRUCTURAL"
    elif "china" in bn:
        theme_primary = "TRADE_GEO"
        policy_axis = "TRADE_CHINA"
    elif "tariffs" in bn or "trade_policy" in bn:
        theme_primary = "TRADE_GEO"
        policy_axis = "TRADE_TARIFFS"
    elif "trump_truth_social" in bn:
        theme_primary = "TRADE_GEO"
        policy_axis = "TRUMP_SOCIAL"
        is_trump_related = True

    # Additional trump flag if text mentions Trump
    if "trump" in text.lower():
        is_trump_related = True

    return {
        "theme_primary": theme_primary or "IDIOSYNCRATIC",
        "is_trump_related": is_trump_related,
        "policy_axis": policy_axis,
        "horizon": horizon,
    }


def build_news_rows(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw ScrapeCreators items into the canonical schema for
    raw.scrapecreators_news_buckets.

    Expected incoming keys per item (from vendor integration):
      - article_id (str)
      - headline (str)
      - content (str)
      - source (str)
      - source_trust_score (float, optional)
      - published_at (datetime/ISO date string)
    """
    rows: List[Dict[str, Any]] = []
    for it in items:
        article_id = it.get("article_id")
        headline = it.get("headline", "") or ""
        content = it.get("content", "") or ""
        source = it.get("source", "")
        source_trust_score = it.get("source_trust_score")
        published_at = it.get("published_at") or it.get("date") or datetime.utcnow()

        if not article_id:
            # Skip items with no stable ID
            continue

        text = f"{headline}\n\n{content}".strip()
        if not text:
            continue

        bucket_name = it.get("bucket_name", "unspecified")
        bucket_meta = map_theme_and_policy(bucket_name, text)

        sentiment = calculate_sentiment_finbert(text, bucket_meta["theme_primary"])

        row = {
            "date": pd.to_datetime(published_at).date(),
            "article_id": str(article_id),
            "theme_primary": bucket_meta["theme_primary"],
            "is_trump_related": bool(bucket_meta["is_trump_related"]),
            "policy_axis": bucket_meta["policy_axis"],
            "horizon": bucket_meta["horizon"],
            "zl_sentiment": sentiment["sentiment"],
            "impact_magnitude": None,  # placeholder until impact logic is wired
            "sentiment_confidence": float(sentiment["confidence"]),
            "sentiment_raw_score": float(sentiment["raw_score"]),
            "headline": headline,
            "content": content,
            "source": source,
            "source_trust_score": float(source_trust_score) if source_trust_score is not None else None,
            "created_at": datetime.utcnow(),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "date", "article_id", "theme_primary", "is_trump_related", "policy_axis",
            "horizon", "zl_sentiment", "impact_magnitude", "sentiment_confidence",
            "sentiment_raw_score", "headline", "content", "source",
            "source_trust_score", "created_at",
        ])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_to_staging_and_merge(df: pd.DataFrame) -> None:
    """
    Write canonical rows to raw_staging.scrapecreators_news_<run_id> and
    MERGE into raw.scrapecreators_news_buckets keyed by article_id (idempotent).
    """
    if df.empty:
        print("[scrapecreators] No news items to load.")
        return

    client = bigquery.Client(project=PROJECT_ID)
    ensure_staging_dataset(client)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    staging_table_id = f"{STAGING_DATASET}.scrapecreators_news_{run_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("article_id", "STRING"),
            bigquery.SchemaField("theme_primary", "STRING"),
            bigquery.SchemaField("is_trump_related", "BOOL"),
            bigquery.SchemaField("policy_axis", "STRING"),
            bigquery.SchemaField("horizon", "STRING"),
            bigquery.SchemaField("zl_sentiment", "STRING"),
            bigquery.SchemaField("impact_magnitude", "STRING"),
            bigquery.SchemaField("sentiment_confidence", "FLOAT"),
            bigquery.SchemaField("sentiment_raw_score", "FLOAT"),
            bigquery.SchemaField("headline", "STRING"),
            bigquery.SchemaField("content", "STRING"),
            bigquery.SchemaField("source", "STRING"),
            bigquery.SchemaField("source_trust_score", "FLOAT"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ],
    )

    print(f"[scrapecreators] Loading {len(df):,} rows into staging table {staging_table_id}...")
    load_job = client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
    load_job.result()
    print(f"[scrapecreators] ✅ Loaded {load_job.output_rows:,} rows into {staging_table_id}")

    # MERGE into canonical raw table keyed on article_id (idempotent)
    merge_sql = f"""
    MERGE `{RAW_TABLE}` T
    USING `{staging_table_id}` S
    ON T.article_id = S.article_id
    WHEN NOT MATCHED THEN
      INSERT (
        date, article_id, theme_primary, is_trump_related, policy_axis,
        horizon, zl_sentiment, impact_magnitude, sentiment_confidence,
        sentiment_raw_score, headline, content, source,
        source_trust_score, created_at
      )
      VALUES (
        S.date, S.article_id, S.theme_primary, S.is_trump_related, S.policy_axis,
        S.horizon, S.zl_sentiment, S.impact_magnitude, S.sentiment_confidence,
        S.sentiment_raw_score, S.headline, S.content, S.source,
        S.source_trust_score, S.created_at
      )
    """
    print(f"[scrapecreators] Merging staging data into {RAW_TABLE}...")
    client.query(merge_sql).result()
    print(f"[scrapecreators] ✅ MERGE complete into {RAW_TABLE}")


def main():
    print("[scrapecreators] Starting news buckets ingestion...")
    items = fetch_all_buckets()
    df = build_news_rows(items)
    load_to_staging_and_merge(df)
    print("[scrapecreators] Ingestion complete.")


if __name__ == "__main__":
    main()






