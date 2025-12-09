#!/usr/bin/env python3
"""
ScrapeCreators News Ingestion - Biofuel Policy Bucket

Fetches biofuel/RFS/EPA policy news from ScrapeCreators API.
Keywords: EPA, RFS, biodiesel, RIN, renewable fuel, LCFS, 45Z
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests

BUCKET_NAME = "biofuel_policy"
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")

# ScrapeCreators uses Google Search API for news
GOOGLE_SEARCH_ENDPOINT = "https://api.scrapecreators.com/v1/google/search"


def fetch_bucket_items() -> List[Dict[str, Any]]:
    """
    Fetch biofuel policy news items using ScrapeCreators Google Search API.

    Returns:
        List of dicts with keys: article_id, headline, content, source, published_at
    """
    if not SCRAPECREATORS_API_KEY:
        raise RuntimeError("SCRAPECREATORS_API_KEY not set. Cannot fetch real data.")

    headers = {"x-api-key": SCRAPECREATORS_API_KEY}

    # Search queries for biofuel policy news
    search_queries = [
        "EPA RFS renewable fuel standard",
        "biodiesel RIN prices D4 D6",
        "LCFS California low carbon fuel",
        "45Z clean fuel production credit",
        "renewable diesel mandate",
    ]

    all_items = []

    for query in search_queries:
        params = {
            "q": query,
            "num": 20,  # Results per query
            "tbm": "nws",  # News search
        }

        try:
            response = requests.get(
                GOOGLE_SEARCH_ENDPOINT, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("organic_results", [])

            # Normalize to expected schema
            for result in results:
                all_items.append(
                    {
                        "article_id": result.get("link", "")
                        .replace("https://", "")
                        .replace("/", "_")[:64],
                        "headline": result.get("title", ""),
                        "content": result.get("snippet", ""),
                        "source": result.get("source", "Google News"),
                        "published_at": result.get("date")
                        or datetime.utcnow().isoformat(),
                        "source_trust_score": 0.80,  # News aggregator trust
                    }
                )

        except Exception as e:
            print(f"Error fetching query '{query}': {e}")
            continue

    return all_items


if __name__ == "__main__":
    items = fetch_bucket_items()
    print(f"Fetched {len(items)} biofuel policy items")
