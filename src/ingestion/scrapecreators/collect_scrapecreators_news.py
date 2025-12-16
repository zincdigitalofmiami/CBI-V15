#!/usr/bin/env python3
"""
ScrapeCreators News Collection
Pulls data from Twitter, Google Search, Facebook, and Truth Social
Target: raw.scrapecreators_news_buckets

API Endpoints:
- Twitter: /v1/twitter/user-tweets (handle param)
- Google: /v1/google/search (query param)
- Facebook: /v1/facebook/profile (url param) - profile only
- Truth Social: /v1/truthsocial/profile (handle param) - profile only, NO posts

Usage:
    python collect_scrapecreators_news.py

Environment Variables Required:
    MOTHERDUCK_TOKEN
    SCRAPECREATORS_API_KEY
"""

import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List

import duckdb
import pandas as pd
import requests

# Config
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")

# Keywords for bucket classification
AG_KEYWORDS = [
    "china",
    "tariff",
    "trade",
    "soybean",
    "agriculture",
    "farmer",
    "biofuel",
    "ethanol",
    "biodiesel",
    "epa",
    "rfs",
    "energy",
    "brazil",
    "mexico",
    "export",
    "import",
    "tax",
    "economy",
    "oil",
    "gas",
    "corn",
]

BIOFUEL_KEYWORDS = ["biofuel", "biodiesel", "ethanol", "rin", "rfs", "renewable"]
ENERGY_KEYWORDS = ["crude", "oil price", "gasoline", "diesel", "opec"]
TARIFF_KEYWORDS = ["tariff", "trade war", "section 301", "china trade"]


def classify_bucket(text: str) -> str:
    """Classify text into Big 8 bucket."""
    text_lower = text.lower()

    if any(kw in text_lower for kw in TARIFF_KEYWORDS):
        return "tariff"
    if any(kw in text_lower for kw in BIOFUEL_KEYWORDS):
        return "biofuel"
    if any(kw in text_lower for kw in ENERGY_KEYWORDS):
        return "energy"
    if any(kw in text_lower for kw in AG_KEYWORDS):
        return "general"
    return "news"


def fetch_twitter_tweets(
    handle: str = "realDonaldTrump", limit: int = 50
) -> List[Dict[str, Any]]:
    """Fetch tweets from Twitter user."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    url = "https://api.scrapecreators.com/v1/twitter/user-tweets"
    params = {"handle": handle, "limit": limit}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        tweets = data.get("tweets", [])

        rows = []
        for tweet in tweets:
            text = tweet.get("full_text", tweet.get("text", ""))
            tweet_id = tweet.get("id_str", tweet.get("id", ""))
            created = tweet.get("created_at")

            bucket = classify_bucket(text)
            is_relevant = bucket != "news"

            rows.append(
                {
                    "article_id": hashlib.md5(
                        f"twitter_{tweet_id}".encode()
                    ).hexdigest(),
                    "date": datetime.now().date(),
                    "published_at": (
                        pd.to_datetime(created) if created else datetime.now()
                    ),
                    "headline": text[:200] if text else "",
                    "content": text,
                    "url": f"https://twitter.com/{handle}/status/{tweet_id}",
                    "author": handle,
                    "bucket_name": bucket,
                    "edition_type": "tweet",
                    "source": "twitter",
                    "source_trust_score": 0.90,
                    "sentiment_score": 0.0,
                    "zl_sentiment": "neutral",
                    "is_trump_related": "trump" in handle.lower(),
                    "policy_axis": "trade" if is_relevant else None,
                    "created_at": datetime.now(),
                }
            )

        print(f"[Twitter] Fetched {len(rows)} tweets from @{handle}")
        return rows

    except Exception as e:
        print(f"[Twitter] Error: {e}")
        return []


def fetch_google_search(queries: List[str]) -> List[Dict[str, Any]]:
    """Fetch Google search results for multiple queries."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    url = "https://api.scrapecreators.com/v1/google/search"

    rows = []
    for query in queries:
        try:
            resp = requests.get(
                url, headers=headers, params={"query": query}, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("organic_results", data.get("results", []))

            for result in results[:5]:  # Top 5 per query
                title = result.get("title", "")
                snippet = result.get("snippet", result.get("description", ""))
                link = result.get("link", result.get("url", ""))

                combined = title + " " + snippet
                bucket = classify_bucket(combined)

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
                        "is_trump_related": "trump" in combined.lower(),
                        "policy_axis": (
                            "trade" if "tariff" in combined.lower() else None
                        ),
                        "created_at": datetime.now(),
                    }
                )

            print(f"[Google] Query '{query[:30]}...': {len(results)} results")

        except Exception as e:
            print(f"[Google] Error for '{query[:20]}...': {e}")

    return rows


def fetch_truthsocial_profile(handle: str = "realDonaldTrump") -> Dict[str, Any]:
    """
    Fetch Truth Social profile (no posts endpoint available).
    Returns profile stats only.
    """
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    url = "https://api.scrapecreators.com/v1/truthsocial/profile"
    params = {"handle": handle}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        print(
            f"[Truth Social] @{data.get('username')}: {data.get('followers_count')} followers, {data.get('statuses_count')} posts"
        )
        return data

    except Exception as e:
        print(f"[Truth Social] Error: {e}")
        return {}


def load_to_motherduck(rows: List[Dict[str, Any]]) -> int:
    """Load rows to raw.scrapecreators_news_buckets."""
    if not rows:
        print("No data to load")
        return 0

    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    df = pd.DataFrame(rows)
    con.register("news_data", df)

    con.execute(
        """
        INSERT INTO raw.scrapecreators_news_buckets 
        SELECT * FROM news_data
        ON CONFLICT (article_id) DO NOTHING
    """
    )

    count = con.execute(
        "SELECT COUNT(*) FROM raw.scrapecreators_news_buckets"
    ).fetchone()[0]
    return count


def main():
    """Main collection routine."""
    print("=" * 80)
    print("SCRAPECREATORS NEWS COLLECTION")
    print("=" * 80)

    if not SCRAPECREATORS_API_KEY:
        raise ValueError("SCRAPECREATORS_API_KEY required")

    all_rows = []

    # 1. Twitter
    twitter_rows = fetch_twitter_tweets("realDonaldTrump", limit=50)
    all_rows.extend(twitter_rows)

    # 2. Google Search
    search_queries = [
        "trump china tariffs soybeans agriculture",
        "biodiesel renewable fuel standard EPA",
        "soybean oil futures market outlook",
        "USDA export sales soybeans",
    ]
    google_rows = fetch_google_search(search_queries)
    all_rows.extend(google_rows)

    # 3. Truth Social profile (no posts available)
    fetch_truthsocial_profile("realDonaldTrump")

    # Load to MotherDuck
    print("\n" + "=" * 80)
    print(f"Total rows: {len(all_rows)}")

    count = load_to_motherduck(all_rows)
    print(f"✅ raw.scrapecreators_news_buckets: {count} total rows")


if __name__ == "__main__":
    main()
