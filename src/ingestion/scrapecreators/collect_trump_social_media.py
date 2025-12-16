#!/usr/bin/env python3
"""
Trump Social Media Tracker - Dedicated Trump Posts Only
Collects Trump posts from Twitter, Facebook, Truth Social
Target: raw.scrapecreators_trump (dedicated Trump table)

Purpose: Track Trump's social media activity for correlation with:
- VIX volatility
- ZL (soybean oil) prices
- Tariff/trade policy sentiment
- Market reactions

Usage:
    python collect_trump_social_media.py
"""

import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List

import duckdb
import pandas as pd
import requests

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")

# ScrapeCreators endpoints
TWITTER_ENDPOINT = "https://api.scrapecreators.com/v1/twitter/user-tweets"
FACEBOOK_ENDPOINT = "https://api.scrapecreators.com/v1/facebook/profile"
TRUTHSOCIAL_ENDPOINT = "https://api.scrapecreators.com/v1/truthsocial/profile"


def fetch_trump_twitter(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch Trump's Twitter/X posts."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    params = {"handle": "realDonaldTrump", "limit": limit}

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

            rows.append(
                {
                    "post_id": hashlib.md5(f"twitter_{tweet_id}".encode()).hexdigest(),
                    "published_date": (
                        pd.to_datetime(created) if created else datetime.now()
                    ),
                    "platform": "twitter",
                    "content": text,
                    "sentiment_score": None,  # Calculate later
                    "zl_impact_score": None,  # Calculate later
                    "source": "scrapecreators_twitter",
                    "ingested_at": datetime.now(),
                }
            )

        print(f"[Twitter] Fetched {len(rows)} Trump tweets")
        return rows

    except Exception as e:
        print(f"[Twitter] Error: {e}")
        return []


def fetch_trump_facebook() -> Dict[str, Any]:
    """Fetch Trump's Facebook profile (posts not available via API)."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    params = {"url": "https://www.facebook.com/DonaldTrump"}

    try:
        resp = requests.get(
            FACEBOOK_ENDPOINT, headers=headers, params=params, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        print(
            f"[Facebook] Profile: {data.get('name')} - {data.get('followers')} followers"
        )
        return data

    except Exception as e:
        print(f"[Facebook] Error: {e}")
        return {}


def fetch_trump_truthsocial() -> Dict[str, Any]:
    """Fetch Trump's Truth Social profile."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    params = {"handle": "realDonaldTrump"}

    try:
        resp = requests.get(
            TRUTHSOCIAL_ENDPOINT, headers=headers, params=params, timeout=60
        )
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
    """Load Trump posts to raw.scrapecreators_trump."""
    if not rows:
        print("No Trump posts to load")
        return 0

    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    df = pd.DataFrame(rows)
    con.register("trump_posts", df)

    # Insert with ON CONFLICT DO NOTHING (idempotent)
    con.execute(
        """
        INSERT INTO raw.scrapecreators_trump 
        SELECT * FROM trump_posts
        ON CONFLICT (post_id) DO NOTHING
    """
    )

    count = con.execute("SELECT COUNT(*) FROM raw.scrapecreators_trump").fetchone()[0]
    con.close()

    return count


def main():
    """Main collection routine."""
    print("=" * 80)
    print("TRUMP SOCIAL MEDIA TRACKER")
    print("=" * 80)

    if not SCRAPECREATORS_API_KEY:
        raise ValueError("SCRAPECREATORS_API_KEY required")

    all_posts = []

    # 1. Twitter (primary source)
    twitter_posts = fetch_trump_twitter(limit=100)
    all_posts.extend(twitter_posts)

    # 2. Facebook profile (no posts via API)
    fetch_trump_facebook()

    # 3. Truth Social profile (no posts via API)
    fetch_trump_truthsocial()

    # Load to MotherDuck
    print("\n" + "=" * 80)
    print(f"Total Trump posts collected: {len(all_posts)}")

    if all_posts:
        count = load_to_motherduck(all_posts)
        print(f"✅ raw.scrapecreators_trump: {count} total rows")
    else:
        print("⚠️  No posts collected (API may only return profiles, not posts)")


if __name__ == "__main__":
    main()
