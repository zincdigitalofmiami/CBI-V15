"""
Tariffs & Trade Policy Bucket Collector.
Fetches tariff/trade policy news from ScrapeCreators API.
"""

import os
import requests
from typing import Any, Dict, List

SCRAPECREATOR_API_KEY = os.getenv("SCRAPECREATOR_API_KEY")
BUCKET_ENDPOINT = os.getenv(
    "SCRAPECREATOR_TARIFFS_ENDPOINT",
    "https://api.scrapecreator.com/v1/tariffs_trade_policy",
)


def fetch_bucket_items() -> List[Dict[str, Any]]:
    """
    Fetch tariffs/trade policy news items from ScrapeCreators.

    Returns:
        List of dicts with keys: article_id, headline, content, source, published_at
    """
    if not SCRAPECREATOR_API_KEY:
        raise RuntimeError("SCRAPECREATOR_API_KEY not set. Cannot fetch real data.")

    headers = {"Authorization": f"Bearer {SCRAPECREATOR_API_KEY}"}

    response = requests.get(BUCKET_ENDPOINT, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()

    items = []
    for item in data.get("items", []):
        items.append(
            {
                "article_id": item.get("id"),
                "headline": item.get("title") or item.get("headline"),
                "content": item.get("body") or item.get("content"),
                "source": item.get("source"),
                "published_at": item.get("published_at") or item.get("date"),
            }
        )

    return items
