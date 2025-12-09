#!/usr/bin/env python3
"""
Master Bucket News Orchestrator

Runs all bucket-level news collectors:
1. ProFarmer Anchor (premium curated)
2. China bucket (Agrimoney, CONAB, Reuters)
3. Tariff bucket (Immigration, Farm Bureau, State Ag)
4. Biofuel bucket
5. Weather bucket
6. Energy bucket
7. Crush bucket
8. FX bucket
9. Fed bucket

Combines with ScrapeCreators API collectors for comprehensive coverage.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import bucket collectors
from src.ingestion.buckets.news.profarmer_anchor import fetch_profarmer_articles
from src.ingestion.buckets.china.collect_china_news import fetch_china_bucket_news
from src.ingestion.buckets.tariff.collect_tariff_news import fetch_tariff_bucket_news

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
RAW_TABLE = "raw.bucket_news"  # Separate table from scrapecreators_news_buckets


def fetch_all_bucket_news() -> List[Dict[str, Any]]:
    """Fetch news from all bucket collectors"""
    all_articles = []
    
    print("\n" + "="*60)
    print("BUCKET NEWS COLLECTION")
    print("="*60)
    
    # ProFarmer Anchor (premium curated)
    print("\n[1/3] ProFarmer Anchor...")
    try:
        profarmer_articles = fetch_profarmer_articles(days_back=7)
        all_articles.extend(profarmer_articles)
    except Exception as e:
        print(f"❌ ProFarmer error: {e}")
    
    # China bucket
    print("\n[2/3] China Bucket...")
    try:
        china_articles = fetch_china_bucket_news()
        all_articles.extend(china_articles)
    except Exception as e:
        print(f"❌ China bucket error: {e}")
    
    # Tariff bucket
    print("\n[3/3] Tariff Bucket...")
    try:
        tariff_articles = fetch_tariff_bucket_news()
        all_articles.extend(tariff_articles)
    except Exception as e:
        print(f"❌ Tariff bucket error: {e}")
    
    # TODO: Add remaining buckets
    # - Biofuel bucket
    # - Weather bucket
    # - Energy bucket
    # - Crush bucket
    # - FX bucket
    # - Fed bucket
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_articles)} articles from all buckets")
    print(f"{'='*60}")
    
    return all_articles


def deduplicate_by_url(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate articles by URL, keeping highest trust score"""
    url_to_article = {}
    
    for article in articles:
        url = article.get("url", "")
        if not url:
            continue
        
        if url in url_to_article:
            existing_trust = url_to_article[url].get("source_trust_score", 0.0)
            new_trust = article.get("source_trust_score", 0.0)
            
            if new_trust > existing_trust:
                url_to_article[url] = article
        else:
            url_to_article[url] = article
    
    unique_articles = list(url_to_article.values())
    print(f"[dedup] Reduced {len(articles)} to {len(unique_articles)} unique articles")
    return unique_articles


def load_to_motherduck(articles: List[Dict[str, Any]]) -> None:
    """Load articles to MotherDuck raw.bucket_news table"""
    if not articles:
        print("[motherduck] No articles to load")
        return
    
    if not MOTHERDUCK_TOKEN:
        raise RuntimeError("MOTHERDUCK_TOKEN not set")
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Add metadata
    df['date'] = pd.to_datetime(df['published_at']).dt.date
    df['created_at'] = datetime.utcnow()
    
    # Connect to MotherDuck
    conn_str = f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}"
    
    with duckdb.connect(conn_str) as conn:
        # Create temp staging table
        conn.execute("CREATE TEMP TABLE staging_bucket_news AS SELECT * FROM df")
        
        # Insert with deduplication
        merge_sql = f"""
        INSERT OR IGNORE INTO {RAW_TABLE}
        SELECT * FROM staging_bucket_news
        """
        
        result = conn.execute(merge_sql)
        rows_inserted = result.fetchone()[0] if result else 0
        
        print(f"[motherduck] ✅ Inserted {rows_inserted:,} new articles into {RAW_TABLE}")


def main():
    """Main orchestrator"""
    print("\n" + "="*60)
    print("BUCKET NEWS INGESTION PIPELINE")
    print("="*60)
    
    # Fetch all articles
    articles = fetch_all_bucket_news()
    
    if not articles:
        print("\n⚠️  No articles fetched")
        return
    
    # Deduplicate
    articles = deduplicate_by_url(articles)
    
    # Load to MotherDuck
    try:
        load_to_motherduck(articles)
        print("\n✅ Pipeline complete")
    except Exception as e:
        print(f"\n❌ Error loading to MotherDuck: {e}")
        raise


if __name__ == "__main__":
    main()

