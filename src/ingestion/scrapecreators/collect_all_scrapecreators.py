#!/usr/bin/env python3
"""
ScrapeCreators Master Collector - EXPANDED
Collects ALL available data from ScrapeCreators API
Runs daily to maximize data collection

Collects:
1. Trump Twitter (100 tweets)
2. Key ag Twitter accounts (USDA, Farm Bureau, etc.)
3. Google News for all Big 8 buckets (expanded queries)
4. Reddit agriculture discussions
5. News aggregation

Target: raw.scrapecreators_news_buckets + raw.scrapecreators_trump
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
GOOGLE_ENDPOINT = "https://api.scrapecreators.com/v1/google/search"
REDDIT_ENDPOINT = "https://api.scrapecreators.com/v1/reddit/search"


# ============================================================================
# EXPANDED TWITTER ACCOUNTS (Agriculture/Trade/Policy)
# ============================================================================
TWITTER_ACCOUNTS = [
    # Political
    "realDonaldTrump",
    "POTUS",
    "SecretaryPete",  # Transportation (logistics)
    # USDA/Government
    "USDA",
    "USDAForeignAg",
    "USDAERS",  # Economic Research
    "USTradeRep",  # USTR
    # Industry
    "FarmBureau",
    "NationalCorn",
    "ASA_Soybeans",
    "GrowthEnergy",  # Ethanol
    "BiodieselBoard",
    # Market/Analysis
    "AgWeb",
    "ProFarmer",
    "DTNCommodities",
    "GrainStats",
]


# ============================================================================
# EXPANDED GOOGLE SEARCH QUERIES (All Big 8 Buckets)
# ============================================================================
GOOGLE_QUERIES = {
    "crush": [
        "soybean crush spread margins",
        "soybean oil meal ratio",
        "NOPA crush report",
        "soybean processing capacity",
    ],
    "china": [
        "China soybean imports demand",
        "Sinograin COFCO purchases",
        "China crushing margins",
        "China vegetable oil stocks",
        "China African swine fever",
        "China hog feed demand",
    ],
    "fx": [
        "Brazilian real currency soybeans",
        "dollar index commodities",
        "Argentina peso soybeans",
        "Mexico peso corn",
    ],
    "fed": [
        "Federal Reserve interest rates agriculture",
        "yield curve commodities",
        "financial conditions index",
        "NFCI agriculture",
    ],
    "tariff": [
        "Section 301 tariffs agriculture",
        "USTR trade policy China",
        "China retaliation tariffs soybeans",
        "WTO agriculture dispute",
        "Trump tariffs trade war",
        "trade negotiations China soybeans",
    ],
    "biofuel": [
        "EPA RFS renewable fuel standard",
        "biodiesel RIN prices D4 D6",
        "LCFS California low carbon fuel",
        "45Z clean fuel production credit",
        "renewable diesel mandate",
        "EPA small refinery exemptions",
    ],
    "energy": [
        "crude oil prices soybeans",
        "heating oil diesel prices",
        "crack spread refining margins",
        "OPEC production soybeans",
    ],
    "volatility": [
        "VIX volatility commodities",
        "market stress agriculture",
        "STLFSI financial stress",
        "commodity volatility soybeans",
    ],
}


# ============================================================================
# REDDIT SUBREDDITS (Agriculture/Trading)
# ============================================================================
REDDIT_SUBREDDITS = [
    "farming",
    "agriculture",
    "commodities",
    "options",  # Trading
    "wallstreetbets",  # Sentiment
]


def classify_bucket(text: str) -> str:
    """Classify text into Big 8 bucket."""
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


def fetch_twitter_account(handle: str, limit: int = 50) -> List[Dict[str, Any]]:
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

            bucket = classify_bucket(text)

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
                    "bucket_name": bucket,
                    "edition_type": "tweet",
                    "source": "twitter",
                    "source_trust_score": (
                        0.90 if handle in ["USDA", "USDAForeignAg"] else 0.75
                    ),
                    "sentiment_score": 0.0,
                    "zl_sentiment": "neutral",
                    "is_trump_related": "trump" in handle.lower(),
                    "policy_axis": None,
                    "created_at": datetime.now(),
                }
            )

        print(f"[Twitter] @{handle}: {len(rows)} tweets")
        return rows

    except Exception as e:
        print(f"[Twitter] @{handle} Error: {e}")
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
                        "policy_axis": None,
                        "created_at": datetime.now(),
                    }
                )

            print(f"[Google] {bucket} - '{query[:30]}...': {len(results)} results")

        except Exception as e:
            print(f"[Google] {bucket} - '{query[:30]}...' Error: {e}")

    return rows


def fetch_reddit_discussions(
    subreddit: str, query: str = "soybeans"
) -> List[Dict[str, Any]]:
    """Fetch Reddit discussions (if endpoint available)."""
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    params = {"subreddit": subreddit, "query": query, "limit": 20}

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

            bucket = classify_bucket(title + " " + body)

            rows.append(
                {
                    "article_id": hashlib.md5(f"reddit_{post_id}".encode()).hexdigest(),
                    "date": datetime.now().date(),
                    "published_at": datetime.now(),
                    "headline": title[:200],
                    "content": body[:500],
                    "url": f"https://reddit.com/r/{subreddit}/comments/{post_id}",
                    "author": post.get("author", ""),
                    "bucket_name": bucket,
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

        print(f"[Reddit] r/{subreddit}: {len(rows)} posts")
        return rows

    except Exception as e:
        print(f"[Reddit] r/{subreddit} Error: {e}")
        return []


def load_to_motherduck(rows: List[Dict[str, Any]], table: str) -> int:
    """Load rows to MotherDuck."""
    if not rows:
        return 0

    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    df = pd.DataFrame(rows)
    con.register("temp_data", df)

    con.execute(
        f"""
        INSERT INTO raw.{table}
        SELECT * FROM temp_data
        ON CONFLICT (article_id) DO NOTHING
    """
    )

    count = con.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
    con.close()

    return count


def main():
    """Main collection routine - EXPANDED."""
    print("=" * 80)
    print("SCRAPECREATORS MASTER COLLECTOR (EXPANDED)")
    print("=" * 80)

    if not SCRAPECREATORS_API_KEY:
        raise ValueError("SCRAPECREATORS_API_KEY required")

    all_rows = []
    trump_rows = []

    # 1. Twitter accounts (18 accounts × 50 tweets = 900 potential tweets)
    print("\n[1/3] Collecting Twitter accounts...")
    for handle in TWITTER_ACCOUNTS:
        tweets = fetch_twitter_account(handle, limit=50)

        if handle == "realDonaldTrump":
            # Separate Trump posts
            trump_rows.extend(tweets)
        else:
            all_rows.extend(tweets)

    # 2. Google searches (8 buckets × 4-6 queries × 10 results = 320-480 articles)
    print("\n[2/3] Collecting Google News...")
    for bucket, queries in GOOGLE_QUERIES.items():
        results = fetch_google_queries(bucket, queries)
        all_rows.extend(results)

    # 3. Reddit (5 subreddits × 20 posts = 100 posts)
    print("\n[3/3] Collecting Reddit discussions...")
    for subreddit in REDDIT_SUBREDDITS:
        posts = fetch_reddit_discussions(subreddit, query="soybeans OR agriculture")
        all_rows.extend(posts)

    # Load to MotherDuck
    print("\n" + "=" * 80)
    print(f"Total items collected: {len(all_rows) + len(trump_rows)}")
    print(f"  General news: {len(all_rows)}")
    print(f"  Trump posts: {len(trump_rows)}")

    if all_rows:
        count = load_to_motherduck(all_rows, "scrapecreators_news_buckets")
        print(f"✅ raw.scrapecreators_news_buckets: {count:,} total rows")

    if trump_rows:
        # Convert Trump tweets to trump table format
        trump_formatted = []
        for row in trump_rows:
            trump_formatted.append(
                {
                    "post_id": row["article_id"],
                    "published_date": row["published_at"],
                    "platform": "twitter",
                    "content": row["content"],
                    "sentiment_score": None,
                    "zl_impact_score": None,
                    "source": "scrapecreators_twitter",
                    "ingested_at": datetime.now(),
                }
            )

        count = load_to_motherduck(trump_formatted, "scrapecreators_trump")
        print(f"✅ raw.scrapecreators_trump: {count:,} total rows")


if __name__ == "__main__":
    main()
