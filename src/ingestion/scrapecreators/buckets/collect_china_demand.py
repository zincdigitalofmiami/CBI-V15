#!/usr/bin/env python3
"""
ScrapeCreators News Ingestion - China Demand Bucket

Fetches China soybean demand, import, trade news.
Keywords: China, Sinograin, COFCO, crushing margins, imports
"""

import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List

import requests

BUCKET_NAME = "china_demand"
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")
GOOGLE_SEARCH_ENDPOINT = "https://api.scrapecreators.com/v1/google/search"


def fetch_bucket_items() -> List[Dict[str, Any]]:
    """
    Fetch China demand news using ScrapeCreators Google Search API.

    Returns:
        List of dicts with metadata: article_id, headline, content, source, published_at, bucket_name, search_query
    """
    if not SCRAPECREATORS_API_KEY:
        raise RuntimeError("SCRAPECREATORS_API_KEY not set")

    headers = {"x-api-key": SCRAPECREATORS_API_KEY}

    search_queries = [
        "China soybean imports demand",
        "Sinograin COFCO soybean purchases",
        "China crushing margins soybean oil",
        "China vegetable oil stocks inventory",
        "China soybean meal demand hog feed",
        "NDRC China soybean policy",
    ]

    all_items = []
    seen_urls = set()  # Deduplication by URL

    for query in search_queries:
        params = {"q": query, "num": 15, "tbm": "nws", "tbs": "qdr:w"}

        try:
            response = requests.get(
                GOOGLE_SEARCH_ENDPOINT, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("organic_results", [])

            for result in results:
                url = result.get("link", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                article_id = hashlib.md5(url.encode()).hexdigest()

                all_items.append(
                    {
                        "article_id": article_id,
                        "headline": result.get("title", ""),
                        "content": result.get("snippet", ""),
                        "source": result.get("source", "Google News"),
                        "published_at": result.get("date")
                        or datetime.utcnow().isoformat(),
                        "source_trust_score": 0.80,
                        "url": url,
                        "bucket_name": BUCKET_NAME,
                        "search_query": query,
                    }
                )

        except Exception as e:
            print(f"[{BUCKET_NAME}] Error fetching '{query}': {e}")
            continue

    print(f"[{BUCKET_NAME}] Fetched {len(all_items)} unique items")
    return all_items


if __name__ == "__main__":
    items = fetch_bucket_items()
    print(f"Fetched {len(items)} China demand items")
