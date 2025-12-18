#!/usr/bin/env python3
"""
Master Bucket News Orchestrator

Runs all bucket-level news collectors:
1. ProFarmer Anchor (premium curated)
2. TradingEconomics soybeans news (free HTML scrape)
3. Existing ScrapeCreators news already in raw.scrapecreators_news_buckets (optional sync)

Loads a unified feed into raw.bucket_news for downstream consumption.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.usda.profarmer_anchor import fetch_profarmer_articles  # type: ignore
from src.ingestion.usda.tradingeconomics_anchor import scrape_tradingeconomics_news  # type: ignore

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
RAW_TABLE = "raw.bucket_news"


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
    raise RuntimeError(
        f"MotherDuck token required (set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN): {last_error}"
    )


def fetch_all_bucket_news() -> List[Dict[str, Any]]:
    """Fetch news from all bucket collectors"""
    all_articles = []
    
    print("\n" + "="*60)
    print("BUCKET NEWS COLLECTION")
    print("="*60)
    
    # ProFarmer Anchor (premium curated)
    print("\n[1/2] ProFarmer Anchor...")
    try:
        profarmer_articles = fetch_profarmer_articles(days_back=7)
        all_articles.extend(profarmer_articles)
    except Exception as e:
        print(f"❌ ProFarmer error: {e}")
    
    # TradingEconomics soybeans news
    print("\n[2/2] TradingEconomics Soybeans News...")
    try:
        te_articles = scrape_tradingeconomics_news(max_words_per_article=500)
        all_articles.extend(te_articles)
    except Exception as e:
        print(f"❌ TradingEconomics error: {e}")
    
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


def sync_scrapecreators_to_bucket_news(conn: duckdb.DuckDBPyConnection) -> int:
    """
    Copy ScrapeCreators bucketed news into raw.bucket_news for a unified feed.
    """
    try:
        conn.execute(
            """
            INSERT INTO raw.bucket_news (id, date, title, content, url, source, bucket, sentiment_score, ingested_at)
            SELECT
              article_id AS id,
              date,
              headline AS title,
              content,
              url,
              COALESCE(source, 'scrapecreators') AS source,
              bucket_name AS bucket,
              CAST(sentiment_score AS DOUBLE) AS sentiment_score,
              CURRENT_TIMESTAMP AS ingested_at
            FROM raw.scrapecreators_news_buckets sc
            WHERE NOT EXISTS (
              SELECT 1 FROM raw.bucket_news bn WHERE bn.id = sc.article_id
            )
            """
        )
        return 1
    except Exception:
        return 0


def load_to_motherduck(articles: List[Dict[str, Any]]) -> None:
    """Load articles to MotherDuck raw.bucket_news table"""
    if not articles:
        print("[motherduck] No articles to load")
        return

    df = pd.DataFrame(articles)
    if df.empty:
        print("[motherduck] No articles to load")
        return

    df_out = pd.DataFrame(
        {
            "id": df.get("article_id"),
            "date": pd.to_datetime(df.get("published_at"), errors="coerce").dt.date,
            "title": df.get("headline"),
            "content": df.get("content"),
            "url": df.get("url"),
            "source": df.get("source"),
            "bucket": df.get("bucket_name"),
            "sentiment_score": df.get("sentiment_score"),
            "ingested_at": datetime.utcnow(),
        }
    )

    conn = connect_motherduck()
    conn.register("staging_bucket_news", df_out)

    # Upsert by id
    conn.execute(
        """
        DELETE FROM raw.bucket_news
        WHERE id IN (SELECT id FROM staging_bucket_news WHERE id IS NOT NULL)
        """
    )
    conn.execute(
        """
        INSERT INTO raw.bucket_news (id, date, title, content, url, source, bucket, sentiment_score, ingested_at)
        SELECT id, date, title, content, url, source, bucket, CAST(sentiment_score AS DOUBLE), ingested_at
        FROM staging_bucket_news
        WHERE id IS NOT NULL
        """
    )

    sync_scrapecreators_to_bucket_news(conn)
    total = conn.execute("SELECT COUNT(*) FROM raw.bucket_news").fetchone()[0]
    conn.close()

    print(f"[motherduck] ✅ raw.bucket_news total rows now {total:,}")


def main():
    """Main orchestrator"""
    print("\n" + "="*60)
    print("BUCKET NEWS INGESTION PIPELINE")
    print("="*60)

    _load_local_env()
    
    # Fetch all articles
    articles = fetch_all_bucket_news()
    
    if not articles:
        print("\n⚠️  No articles fetched; syncing ScrapeCreators → raw.bucket_news only")
        try:
            conn = connect_motherduck()
            sync_scrapecreators_to_bucket_news(conn)
            total = conn.execute("SELECT COUNT(*) FROM raw.bucket_news").fetchone()[0]
            conn.close()
            print(f"[motherduck] ✅ raw.bucket_news total rows now {total:,}")
        except Exception as e:
            print(f"[motherduck] ❌ Sync failed: {e}")
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
