#!/usr/bin/env python3
"""
ScrapeCreators News Ingestion - Canonical Bucketed Pipeline

Role:
- Pull news items from ScrapeCreators (via per-bucket collectors).
- Segment them into the canonical news bucket schema:
  raw.scrapecreators_news_buckets (theme_primary, is_trump_related, policy_axis, horizon, zl_sentiment, impact_magnitude, etc.).
- Use DuckDB MERGE to keep ingestion idempotent (no duplicates).

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

import duckdb
import pandas as pd

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

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
RAW_TABLE = "raw.scrapecreators_news_buckets"


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
            raise RuntimeError(
                f"Bucket module {module.__name__} missing fetch_bucket_items"
            )

        items = fetch_fn()
        if not isinstance(items, list):
            raise RuntimeError(
                f"{module.__name__}.fetch_bucket_items must return a list of dicts"
            )

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


def deduplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Smart deduplication: Remove duplicates by article_id and URL.
    If same URL appears multiple times, keep the one with highest source_trust_score.
    """
    seen_ids = set()
    url_to_item = {}  # Track best item per URL

    for item in items:
        article_id = item.get("article_id")
        url = item.get("url", "")
        trust_score = item.get("source_trust_score", 0.0)

        # Skip if we've seen this exact article_id
        if article_id in seen_ids:
            continue

        # For URL deduplication, keep highest trust score
        if url:
            if url in url_to_item:
                existing_trust = url_to_item[url].get("source_trust_score", 0.0)
                if trust_score > existing_trust:
                    # Replace with higher trust version
                    old_id = url_to_item[url].get("article_id")
                    seen_ids.discard(old_id)  # Remove old ID
                    url_to_item[url] = item
                    seen_ids.add(article_id)
                # else: keep existing higher-trust version
            else:
                url_to_item[url] = item
                seen_ids.add(article_id)
        else:
            # No URL, just track by article_id
            seen_ids.add(article_id)

    # Return deduplicated items
    unique_items = list(url_to_item.values())
    print(f"[dedup] Reduced {len(items)} items to {len(unique_items)} unique items")
    return unique_items


def enrich_metadata(
    item: Dict[str, Any], bucket_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add smart metadata tags based on content analysis.

    Returns enriched metadata dict with:
    - impact_magnitude: HIGH/MEDIUM/LOW based on keywords
    - horizon: FLASH/TACTICAL/STRUCTURAL based on content
    - additional flags
    """
    text = f"{item.get('headline', '')} {item.get('content', '')}".lower()

    # Impact magnitude keywords
    high_impact_keywords = [
        "breaking",
        "emergency",
        "crisis",
        "ban",
        "embargo",
        "war",
        "collapse",
        "surge",
        "plunge",
        "record",
        "unprecedented",
    ]
    medium_impact_keywords = [
        "increase",
        "decrease",
        "change",
        "policy",
        "regulation",
        "forecast",
        "estimate",
        "report",
        "data",
    ]

    impact_magnitude = "LOW"
    if any(kw in text for kw in high_impact_keywords):
        impact_magnitude = "HIGH"
    elif any(kw in text for kw in medium_impact_keywords):
        impact_magnitude = "MEDIUM"

    # Horizon refinement
    horizon = bucket_meta.get("horizon", "TACTICAL")
    flash_keywords = ["breaking", "just in", "alert", "urgent", "now"]
    structural_keywords = ["long-term", "structural", "mandate", "law", "regulation"]

    if any(kw in text for kw in flash_keywords):
        horizon = "FLASH"
    elif any(kw in text for kw in structural_keywords):
        horizon = "STRUCTURAL"

    return {"impact_magnitude": impact_magnitude, "horizon": horizon}


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
      - url (str, optional)
      - bucket_name (str)
      - search_query (str, optional)
    """
    # First deduplicate
    items = deduplicate_items(items)

    rows: List[Dict[str, Any]] = []
    for it in items:
        article_id = it.get("article_id")
        headline = it.get("headline", "") or ""
        content = it.get("content", "") or ""
        source = it.get("source", "")
        source_trust_score = it.get("source_trust_score", 0.75)
        published_at = it.get("published_at") or it.get("date") or datetime.utcnow()
        url = it.get("url", "")
        search_query = it.get("search_query", "")

        if not article_id:
            # Skip items with no stable ID
            continue

        text = f"{headline}\n\n{content}".strip()
        if not text:
            continue

        bucket_name = it.get("bucket_name", "unspecified")
        bucket_meta = map_theme_and_policy(bucket_name, text)

        # Enrich with smart metadata
        enriched_meta = enrich_metadata(it, bucket_meta)

        sentiment = calculate_sentiment_finbert(text, bucket_meta["theme_primary"])

        row = {
            "date": pd.to_datetime(published_at).date(),
            "article_id": str(article_id),
            "theme_primary": bucket_meta["theme_primary"],
            "is_trump_related": bool(bucket_meta["is_trump_related"]),
            "policy_axis": bucket_meta["policy_axis"],
            "horizon": enriched_meta["horizon"],
            "zl_sentiment": sentiment["sentiment"],
            "impact_magnitude": enriched_meta["impact_magnitude"],
            "sentiment_confidence": float(sentiment["confidence"]),
            "sentiment_raw_score": float(sentiment["raw_score"]),
            "headline": headline,
            "content": content,
            "source": source,
            "source_trust_score": (
                float(source_trust_score) if source_trust_score is not None else None
            ),
            "url": url,  # Track source URL
            "search_query": search_query,  # Track which query found this
            "created_at": datetime.utcnow(),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "article_id",
                "theme_primary",
                "is_trump_related",
                "policy_axis",
                "horizon",
                "zl_sentiment",
                "impact_magnitude",
                "sentiment_confidence",
                "sentiment_raw_score",
                "headline",
                "content",
                "source",
                "source_trust_score",
                "created_at",
            ]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_to_motherduck(df: pd.DataFrame) -> None:
    """
    Load canonical rows to MotherDuck using INSERT OR IGNORE (idempotent).
    """
    if df.empty:
        print("[scrapecreators] No news items to load.")
        return

    if not MOTHERDUCK_TOKEN:
        raise RuntimeError("MOTHERDUCK_TOKEN not set")

    conn_str = f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}"

    print(f"[scrapecreators] Loading {len(df):,} rows to MotherDuck {RAW_TABLE}...")

    with duckdb.connect(conn_str) as conn:
        # Create temp table from dataframe
        conn.execute("CREATE TEMP TABLE staging_news AS SELECT * FROM df")

        # INSERT OR IGNORE to avoid duplicates (keyed on article_id)
        merge_sql = f"""
        INSERT OR IGNORE INTO {RAW_TABLE}
        SELECT * FROM staging_news
        """

        result = conn.execute(merge_sql)
        rows_inserted = result.fetchone()[0] if result else 0

        print(
            f"[scrapecreators] ✅ Inserted {rows_inserted:,} new rows into {RAW_TABLE}"
        )

        # Verify total count
        total = conn.execute(f"SELECT COUNT(*) FROM {RAW_TABLE}").fetchone()[0]
        print(f"[scrapecreators] Total rows in {RAW_TABLE}: {total:,}")


def main():
    print("[scrapecreators] Starting news buckets ingestion...")
    items = fetch_all_buckets()
    df = build_news_rows(items)
    load_to_motherduck(df)
    print("[scrapecreators] Ingestion complete.")


if __name__ == "__main__":
    main()
