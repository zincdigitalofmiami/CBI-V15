#!/usr/bin/env python3
"""
ScrapeCreators Big 8 Bucket Collector
Loads profiles from config/scrapecreators_profiles.yaml
Collects MAXIMUM data for each Big 8 bucket

Run daily to build comprehensive dataset.
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd
import requests
import yaml

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/Volumes/Satechi Hub/CBI-V15")
CONFIG_PATH = PROJECT_ROOT / "config" / "scrapecreators_profiles.yaml"

# Environment
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")

# ScrapeCreators endpoints
TWITTER_ENDPOINT = "https://api.scrapecreators.com/v1/twitter/user-tweets"
GOOGLE_ENDPOINT = "https://api.scrapecreators.com/v1/google/search"
REDDIT_ENDPOINT = "https://api.scrapecreators.com/v1/reddit/search"


def load_config() -> Dict[str, Any]:
    """Load ScrapeCreators profile configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def classify_bucket(text: str, bucket_hint: str = None) -> str:
    """Classify text into Big 8 bucket (with hint from source)."""
    if bucket_hint:
        return bucket_hint

    text_lower = text.lower()

    if any(kw in text_lower for kw in ["tariff", "trade war", "section 301"]):
        return "tariff"
    if any(
        kw in text_lower for kw in ["biofuel", "biodiesel", "ethanol", "rin", "rfs"]
    ):
        return "biofuel"
    if any(kw in text_lower for kw in ["crude", "oil price", "opec"]):
        return "energy"
    if any(kw in text_lower for kw in ["china", "sinograin", "cofco"]):
        return "china"
    if any(kw in text_lower for kw in ["crush", "meal", "processing"]):
        return "crush"
    if any(kw in text_lower for kw in ["real", "peso", "dollar index", "currency"]):
        return "fx"
    if any(kw in text_lower for kw in ["fed", "interest rate", "yield curve"]):
        return "fed"
    if any(kw in text_lower for kw in ["vix", "volatility", "stress"]):
        return "volatility"

    return "general"


def fetch_twitter_account(
    handle: str, bucket: str, limit: int = 100
) -> List[Dict[str, Any]]:
    """Fetch tweets from a Twitter account."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    params = {"handle": handle, "limit": limit}

    try:
        resp = requests.get(
            TWITTER_ENDPOINT, headers=headers, params=params, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        tweets = data.get("tweets", [])

        rows = []
        for tweet in tweets:
            text = tweet.get("full_text", tweet.get("text", ""))
            tweet_id = tweet.get("id_str", tweet.get("id", ""))
            created = tweet.get("created_at")

            # Use bucket hint but allow reclassification
            final_bucket = classify_bucket(text, bucket)

            rows.append(
                {
                    "article_id": hashlib.md5(
                        f"twitter_{handle}_{tweet_id}".encode()
                    ).hexdigest(),
                    "date": datetime.now().date(),
                    "published_at": (
                        pd.to_datetime(created) if created else datetime.now()
                    ),
                    "headline": text[:200] if text else "",
                    "content": text,
                    "url": f"https://twitter.com/{handle}/status/{tweet_id}",
                    "author": handle,
                    "bucket_name": final_bucket,
                    "edition_type": "tweet",
                    "source": "twitter",
                    "source_trust_score": 0.85,
                    "sentiment_score": 0.0,
                    "zl_sentiment": "neutral",
                    "is_trump_related": "trump" in handle.lower(),
                    "policy_axis": (
                        bucket if bucket in ["tariff", "fed", "biofuel"] else None
                    ),
                    "created_at": datetime.now(),
                }
            )

        return rows

    except Exception as e:
        print(f"  ❌ @{handle}: {e}")
        return []


def fetch_google_queries(bucket: str, queries: List[str]) -> List[Dict[str, Any]]:
    """Fetch Google search results for bucket queries."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}

    rows = []
    for query in queries:
        try:
            resp = requests.get(
                GOOGLE_ENDPOINT, headers=headers, params={"query": query}, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", data.get("organic_results", []))

            for result in results[:10]:  # Top 10 per query
                title = result.get("title", "")
                snippet = result.get("snippet", result.get("description", ""))
                link = result.get("link", result.get("url", ""))

                if not link:
                    continue

                rows.append(
                    {
                        "article_id": hashlib.md5(
                            f"google_{link}".encode()
                        ).hexdigest(),
                        "date": datetime.now().date(),
                        "published_at": datetime.now(),
                        "headline": title[:200] if title else "",
                        "content": snippet,
                        "url": link,
                        "author": None,
                        "bucket_name": bucket,
                        "edition_type": "search_result",
                        "source": "google_search",
                        "source_trust_score": 0.70,
                        "sentiment_score": 0.0,
                        "zl_sentiment": "neutral",
                        "is_trump_related": "trump" in (title + snippet).lower(),
                        "policy_axis": (
                            bucket if bucket in ["tariff", "fed", "biofuel"] else None
                        ),
                        "created_at": datetime.now(),
                    }
                )

        except Exception as e:
            print(f"  ❌ Query '{query[:30]}...': {e}")

    return rows


def fetch_reddit_subreddit(
    subreddit: str, bucket: str, query: str = "soybeans"
) -> List[Dict[str, Any]]:
    """Fetch Reddit discussions."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    params = {"subreddit": subreddit, "query": query, "limit": 25}

    try:
        resp = requests.get(REDDIT_ENDPOINT, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get("posts", [])

        rows = []
        for post in posts:
            title = post.get("title", "")
            body = post.get("selftext", "")
            post_id = post.get("id", "")

            final_bucket = classify_bucket(title + " " + body, bucket)

            rows.append(
                {
                    "article_id": hashlib.md5(f"reddit_{post_id}".encode()).hexdigest(),
                    "date": datetime.now().date(),
                    "published_at": datetime.now(),
                    "headline": title[:200],
                    "content": body[:500],
                    "url": f"https://reddit.com/r/{subreddit}/comments/{post_id}",
                    "author": post.get("author", ""),
                    "bucket_name": final_bucket,
                    "edition_type": "reddit_post",
                    "source": "reddit",
                    "source_trust_score": 0.60,
                    "sentiment_score": 0.0,
                    "zl_sentiment": "neutral",
                    "is_trump_related": False,
                    "policy_axis": None,
                    "created_at": datetime.now(),
                }
            )

        return rows

    except Exception as e:
        print(f"  ❌ r/{subreddit}: {e}")
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

    con.execute(
        """
        INSERT INTO raw.scrapecreators_news_buckets
        SELECT * FROM temp_data
        ON CONFLICT (article_id) DO NOTHING
    """
    )

    count = con.execute(
        "SELECT COUNT(*) FROM raw.scrapecreators_news_buckets"
    ).fetchone()[0]
    con.close()

    return count


def main():
    """Main collection routine - Big 8 buckets."""
    print("=" * 80)
    print("SCRAPECREATORS BIG 8 BUCKET COLLECTOR")
    print("=" * 80)

    if not SCRAPECREATORS_API_KEY:
        raise ValueError("SCRAPECREATORS_API_KEY required")

    # Load configuration
    config = load_config()

    all_rows = []
    bucket_stats = {}

    # Process each Big 8 bucket
    big8_buckets = [
        "crush",
        "china",
        "fx",
        "fed",
        "tariff",
        "biofuel",
        "energy",
        "volatility",
    ]

    for bucket in big8_buckets:
        if bucket not in config:
            print(f"\n⚠️  Bucket '{bucket}' not in config, skipping")
            continue

        print(f"\n{'='*80}")
        print(f"BUCKET: {bucket.upper()}")
        print(f"{'='*80}")

        bucket_config = config[bucket]
        bucket_rows = []

        # 1. Twitter accounts
        twitter_accounts = bucket_config.get("twitter_accounts", [])
        if twitter_accounts:
            print(f"\n[Twitter] {len(twitter_accounts)} accounts...")
            for handle in twitter_accounts:
                if not handle or handle.startswith("#"):
                    continue
                tweets = fetch_twitter_account(handle, bucket, limit=100)
                bucket_rows.extend(tweets)
                if tweets:
                    print(f"  ✅ @{handle}: {len(tweets)} tweets")

        # 2. Google queries
        google_queries = bucket_config.get("google_queries", [])
        if google_queries:
            print(f"\n[Google] {len(google_queries)} queries...")
            results = fetch_google_queries(bucket, google_queries)
            bucket_rows.extend(results)
            print(f"  ✅ Total: {len(results)} articles")

        # 3. Reddit subreddits
        reddit_subs = bucket_config.get("reddit_subreddits", [])
        if reddit_subs:
            print(f"\n[Reddit] {len(reddit_subs)} subreddits...")
            for subreddit in reddit_subs:
                posts = fetch_reddit_subreddit(subreddit, bucket)
                bucket_rows.extend(posts)
                if posts:
                    print(f"  ✅ r/{subreddit}: {len(posts)} posts")

        bucket_stats[bucket] = len(bucket_rows)
        all_rows.extend(bucket_rows)
        print(f"\n✅ {bucket.upper()}: {len(bucket_rows)} total items")

    # Load to MotherDuck
    print("\n" + "=" * 80)
    print("LOADING TO MOTHERDUCK")
    print("=" * 80)
    print(f"Total items collected: {len(all_rows)}")

    for bucket, count in bucket_stats.items():
        print(f"  {bucket:12} {count:5,} items")

    if all_rows:
        total_count = load_to_motherduck(all_rows)
        print(f"\n✅ raw.scrapecreators_news_buckets: {total_count:,} total rows")
    else:
        print("\n⚠️  No items collected")


if __name__ == "__main__":
    main()
