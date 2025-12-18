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
import time
from datetime import datetime
from pathlib import Path
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

# Repo paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
RAW_DDL_PATH = PROJECT_ROOT / "database" / "ddl" / "02_raw" / "080_raw_news_articles.sql"


# ============================================================================
# EXPANDED TWITTER ACCOUNTS (Agriculture/Trade/Policy)
# ============================================================================
TWITTER_ACCOUNTS = [
    # Political
    "realDonaldTrump",
    "POTUS",
    "SecretaryPete",  # Transportation (logistics)
    "CommerceGov",
    "USTreasury",
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
    "CleanFuelsNow",
    "RFA_Ethanol",
    "BiodieselMag",
    "EIAgov",
    "CMEGroup",
    "Bunge",
    "ADMupdates",
    "CargillNews",
    "RaboResearch",
    # Palm oil / veg oils
    "MPOBOfficial",
    "GAPKI_ID",
    "WillmarIntl",
    "IOI_Group",
    # Market/Analysis
    "AgWeb",
    "ProFarmer",
    "DTNCommodities",
    "GrainStats",
    # China / trade signalers
    "COFCO_Group",
    "SCMPNews",
    "CGTNOfficial",
    "XHNews",
    "ChinaDaily",
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


def infer_policy_axis(bucket: str, text: str, is_trump_related: bool) -> str | None:
    """
    Map an item to the canonical policy_axis values used by SQL macros.
    Keep this conservative: only emit values we know downstream uses.
    """
    if not is_trump_related:
        return None

    t = text.lower()
    if bucket == "china" or "china" in t or "cofco" in t or "sinograin" in t:
        return "TRADE_CHINA"
    if bucket == "tariff" or "tariff" in t or "trade war" in t or "section 301" in t:
        return "TRADE_TARIFFS"
    return None


def infer_zl_sentiment(bucket: str, text: str, is_trump_related: bool) -> tuple[str, float]:
    """
    Lightweight heuristic sentiment for ZL from text.
    This avoids heavyweight NLP deps in scheduled ingestion.

    Returns:
      (zl_sentiment_label, sentiment_score) where sentiment_score ∈ [-1, 1].
    """
    t = text.lower()

    bullish_terms = [
        "remove tariffs",
        "tariffs lifted",
        "deal reached",
        "agreement reached",
        "phase one",
        "purchase",
        "buy",
        "imports surge",
        "demand strong",
        "mandate",
        "increase rvo",
    ]
    bearish_terms = [
        "tariffs imposed",
        "trade war",
        "retaliation",
        "embargo",
        "ban",
        "export cancelled",
        "weak demand",
        "recession",
    ]

    score = 0.0
    for term in bullish_terms:
        if term in t:
            score += 1.0
    for term in bearish_terms:
        if term in t:
            score -= 1.0

    # Bucket bias: tariff/trade content tends to be risk-off for the soy complex unless explicitly positive.
    if bucket in {"tariff", "china"} and is_trump_related and score == 0.0:
        if any(k in t for k in ["tariff", "trade war", "section 301", "retaliation"]):
            score = -0.5

    if score > 0.0:
        return ("BULLISH_ZL", min(1.0, score / 2.0))
    if score < 0.0:
        return ("BEARISH_ZL", max(-1.0, score / 2.0))
    return ("NEUTRAL", 0.0)


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
            is_trump_related = "trump" in handle.lower() or "trump" in (text or "").lower()
            policy_axis = infer_policy_axis(bucket, text or "", is_trump_related)
            zl_sentiment, sentiment_score = infer_zl_sentiment(bucket, text or "", is_trump_related)

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
                    "sentiment_score": sentiment_score,
                    "zl_sentiment": zl_sentiment,
                    "is_trump_related": is_trump_related,
                    "policy_axis": policy_axis,
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
                if not link:
                    continue
                is_trump_related = "trump" in (title + snippet).lower()
                full_text = f"{title}\n\n{snippet}".strip()
                zl_sentiment, sentiment_score = infer_zl_sentiment(
                    bucket, full_text, is_trump_related
                )

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
                        "sentiment_score": sentiment_score,
                        "zl_sentiment": zl_sentiment,
                        "is_trump_related": is_trump_related,
                        "policy_axis": infer_policy_axis(bucket, full_text, is_trump_related),
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
            is_trump_related = "trump" in (title + " " + body).lower()
            zl_sentiment, sentiment_score = infer_zl_sentiment(
                bucket, title + "\n\n" + body, is_trump_related
            )

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
                    "sentiment_score": sentiment_score,
                    "zl_sentiment": zl_sentiment,
                    "is_trump_related": is_trump_related,
                    "policy_axis": infer_policy_axis(bucket, title + "\n\n" + body, is_trump_related),
                    "created_at": datetime.now(),
                }
            )

        print(f"[Reddit] r/{subreddit}: {len(rows)} posts")
        return rows

    except Exception as e:
        print(f"[Reddit] r/{subreddit} Error: {e}")
        return []


def ensure_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure required ScrapeCreators raw tables exist in the target database."""
    if not RAW_DDL_PATH.exists():
        raise FileNotFoundError(f"Missing required DDL file: {RAW_DDL_PATH}")

    ddl_sql = RAW_DDL_PATH.read_text(encoding="utf-8")
    con.execute(ddl_sql)


def connect_motherduck() -> duckdb.DuckDBPyConnection:
    """Connect to MotherDuck with small retries to handle transient network issues."""
    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
            ensure_raw_tables(con)
            return con
        except Exception as e:
            last_error = e
            sleep_s = min(2**attempt, 20)
            print(f"[motherduck] Connect attempt {attempt}/5 failed: {e}; retrying in {sleep_s}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"Could not connect to MotherDuck after retries: {last_error}")


def load_news_to_motherduck(rows: List[Dict[str, Any]]) -> int:
    """Load rows to MotherDuck raw.scrapecreators_news_buckets (idempotent)."""
    if not rows:
        return 0

    con = connect_motherduck()

    df = pd.DataFrame(rows)
    con.register("temp_data", df)

    con.execute(
        """
        INSERT INTO raw.scrapecreators_news_buckets (
            article_id,
            date,
            published_at,
            headline,
            content,
            url,
            author,
            bucket_name,
            edition_type,
            source,
            source_trust_score,
            sentiment_score,
            zl_sentiment,
            is_trump_related,
            policy_axis,
            created_at
        )
        SELECT
            article_id,
            date,
            published_at,
            headline,
            content,
            url,
            author,
            bucket_name,
            edition_type,
            source,
            source_trust_score,
            sentiment_score,
            zl_sentiment,
            is_trump_related,
            policy_axis,
            created_at
        FROM temp_data
        ON CONFLICT (article_id) DO NOTHING
    """,
    )

    count = con.execute("SELECT COUNT(*) FROM raw.scrapecreators_news_buckets").fetchone()[
        0
    ]
    con.close()
    return count


def load_trump_to_motherduck(rows: List[Dict[str, Any]]) -> int:
    """Load rows to MotherDuck raw.scrapecreators_trump (idempotent)."""
    if not rows:
        return 0

    con = connect_motherduck()

    df = pd.DataFrame(rows)
    con.register("temp_data", df)

    con.execute(
        """
        INSERT INTO raw.scrapecreators_trump (
            post_id,
            published_date,
            platform,
            content,
            sentiment_score,
            zl_impact_score,
            source,
            ingested_at
        )
        SELECT
            post_id,
            published_date,
            platform,
            content,
            sentiment_score,
            zl_impact_score,
            source,
            ingested_at
        FROM temp_data
        ON CONFLICT (post_id) DO NOTHING
    """,
    )

    count = con.execute("SELECT COUNT(*) FROM raw.scrapecreators_trump").fetchone()[0]
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
        count = load_news_to_motherduck(all_rows)
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

        count = load_trump_to_motherduck(trump_formatted)
        print(f"✅ raw.scrapecreators_trump: {count:,} total rows")


if __name__ == "__main__":
    main()
