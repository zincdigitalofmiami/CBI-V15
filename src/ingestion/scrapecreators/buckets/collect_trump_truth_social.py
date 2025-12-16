#!/usr/bin/env python3
"""
ScrapeCreators Social Ingestion - Trump Truth Social Bucket

Fetches Trump's Truth Social posts using ScrapeCreators API.
Focus: Trade, tariffs, China, agriculture, biofuels, energy policy
"""

import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List

import requests

BUCKET_NAME = "trump_truth_social"
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")
TRUTH_SOCIAL_ENDPOINT = "https://api.scrapecreators.com/v1/truthsocial/profile"


def fetch_bucket_items() -> List[Dict[str, Any]]:
    """
    Fetch Trump Truth Social posts using ScrapeCreators API.

    Returns:
        List of dicts with metadata: article_id, headline, content, source, published_at, bucket_name
    """
    if not SCRAPECREATORS_API_KEY:
        raise RuntimeError("SCRAPECREATORS_API_KEY not set")

    headers = {"x-api-key": SCRAPECREATORS_API_KEY}

    # Fetch Trump's recent posts
    params = {"handle": "realDonaldTrump", "limit": 50}  # Last 50 posts

    try:
        response = requests.get(
            TRUTH_SOCIAL_ENDPOINT, headers=headers, params=params, timeout=30
        )
        response.raise_for_status()

        data = response.json()
        posts = data.get("posts", []) or data.get("data", [])

        all_items = []

        # Filter for ag/trade/policy relevant posts
        ag_keywords = [
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
            "mexico",
            "brazil",
            "export",
            "import",
        ]

        for post in posts:
            text = post.get("text", "") or post.get("content", "")
            text_lower = text.lower()

            # Only include posts mentioning ag/trade keywords
            if not any(keyword in text_lower for keyword in ag_keywords):
                continue

            post_id = post.get("id", "")
            article_id = hashlib.md5(f"truthsocial_{post_id}".encode()).hexdigest()

            all_items.append(
                {
                    "article_id": article_id,
                    "headline": (
                        text[:100] + "..." if len(text) > 100 else text
                    ),  # First 100 chars as headline
                    "content": text,
                    "source": "Truth Social - Donald Trump",
                    "published_at": post.get("created_at")
                    or post.get("date")
                    or datetime.utcnow().isoformat(),
                    "source_trust_score": 0.95,  # High trust - direct from source
                    "url": post.get(
                        "url",
                        f"https://truthsocial.com/@realDonaldTrump/posts/{post_id}",
                    ),
                    "bucket_name": BUCKET_NAME,
                    "is_trump_post": True,
                }
            )

        print(
            f"[{BUCKET_NAME}] Fetched {len(all_items)} relevant Trump posts (filtered from {len(posts)} total)"
        )
        return all_items

    except Exception as e:
        print(f"[{BUCKET_NAME}] Error fetching Truth Social: {e}")
        return []


if __name__ == "__main__":
    items = fetch_bucket_items()
    print(f"Fetched {len(items)} Trump Truth Social items")
