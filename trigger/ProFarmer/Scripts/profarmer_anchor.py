#!/usr/bin/env python3
"""
ProFarmer Anchor Scraper - Primary Curated News Feed

Scrapes ProFarmer's daily editions:
- First Thing Today (pre_open)
- Ahead of the Open (pre_open)
- After the Bell (post_close)
- Agriculture News (intraday)
- Newsletters (newsletter)

Requires authentication (credentials in .env).
"""

import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


class ProFarmerScraper:
    """Authenticated scraper for ProFarmer news"""

    BASE_URL = "https://www.profarmer.com"

    SECTIONS = [
        {
            "name": "First Thing Today",
            "url": "/news/first-thing-today",
            "edition_type": "pre_open",
        },
        {
            "name": "Ahead of the Open",
            "url": "/news/ahead-of-the-open",
            "edition_type": "pre_open",
        },
        {
            "name": "After the Bell",
            "url": "/news/after-the-bell",
            "edition_type": "post_close",
        },
        {
            "name": "Agriculture News",
            "url": "/news/agriculture-news",
            "edition_type": "intraday",
        },
        {"name": "Newsletters", "url": "/newsletters", "edition_type": "newsletter"},
    ]

    def __init__(self):
        self.username = os.getenv("PROFARMER_USERNAME")
        self.password = os.getenv("PROFARMER_PASSWORD")

        if not self.username or not self.password:
            raise RuntimeError(
                "PROFARMER_USERNAME and PROFARMER_PASSWORD must be set in .env"
            )

        self.session = requests.Session()
        self._login()

    def _login(self):
        """Authenticate with ProFarmer"""
        login_url = f"{self.BASE_URL}/login"

        # Get login page to extract CSRF token if needed
        response = self.session.get(login_url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")

        # Look for CSRF token (common pattern)
        csrf_token = None
        csrf_input = soup.find("input", {"name": "csrf_token"}) or soup.find(
            "input", {"name": "_token"}
        )
        if csrf_input:
            csrf_token = csrf_input.get("value")

        # Login payload
        payload = {
            "username": self.username,
            "password": self.password,
        }

        if csrf_token:
            payload["csrf_token"] = csrf_token

        # Submit login
        response = self.session.post(login_url, data=payload, timeout=30)
        response.raise_for_status()

        # Verify login success (check for redirect or user menu)
        if "login" in response.url.lower() and "logout" not in response.text.lower():
            raise RuntimeError("ProFarmer login failed - check credentials")

        print("[ProFarmer] ✅ Login successful")

    def scrape_section(
        self, section: Dict[str, str], days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Scrape articles from a ProFarmer section.

        Args:
            section: Dict with name, url, edition_type
            days_back: How many days back to scrape

        Returns:
            List of article dicts
        """
        section_url = f"{self.BASE_URL}{section['url']}"

        response = self.session.get(section_url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        articles = []

        # Find article links (adjust selectors based on actual HTML structure)
        article_links = soup.select("article a.article-link") or soup.select(
            "div.article-item a"
        )

        for link in article_links[:20]:  # Limit to 20 most recent per section
            article_url = link.get("href")
            if not article_url:
                continue

            # Make absolute URL
            if not article_url.startswith("http"):
                article_url = f"{self.BASE_URL}{article_url}"

            # Scrape article detail
            try:
                article = self._scrape_article_detail(
                    article_url, section["edition_type"]
                )
                if article:
                    articles.append(article)
            except Exception as e:
                print(f"[ProFarmer] Error scraping {article_url}: {e}")
                continue

        print(f"[ProFarmer] Scraped {len(articles)} articles from {section['name']}")
        return articles

    def _scrape_article_detail(
        self, url: str, edition_type: str, max_words: int = 500
    ) -> Dict[str, Any]:
        """
        Scrape individual article detail page.

        Args:
            url: Article URL
            edition_type: Edition type (pre_open, post_close, etc.)
            max_words: Limit content to first N words (default 500)
        """
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Extract article components (adjust selectors based on actual HTML)
        title_elem = soup.select_one("h1.article-title") or soup.select_one("h1")
        author_elem = soup.select_one("span.author") or soup.select_one("div.byline")
        date_elem = soup.select_one("time") or soup.select_one("span.date")
        body_elem = soup.select_one("div.article-body") or soup.select_one("article")

        title = title_elem.get_text(strip=True) if title_elem else ""
        author = author_elem.get_text(strip=True) if author_elem else "ProFarmer"
        body = body_elem.get_text(separator=" ", strip=True) if body_elem else ""

        # Limit to first N words
        words = body.split()
        if len(words) > max_words:
            body = " ".join(words[:max_words]) + "..."

        # Extract date
        published_at = datetime.utcnow().isoformat()
        if date_elem and date_elem.has_attr("datetime"):
            published_at = date_elem["datetime"]

        # Generate article_id from URL
        article_id = hashlib.md5(url.encode()).hexdigest()

        return {
            "article_id": article_id,
            "headline": title,
            "content": body,
            "author": author,
            "source": "ProFarmer",
            "source_trust_score": 0.95,  # Premium curated source
            "published_at": published_at,
            "url": url,
            "edition_type": edition_type,
            "bucket_name": "profarmer_anchor",  # Will be mapped to Big 8 buckets downstream
        }


def fetch_profarmer_articles(days_back: int = 7) -> List[Dict[str, Any]]:
    """
    Main entry point: Fetch all ProFarmer articles from all sections.

    Args:
        days_back: How many days back to scrape

    Returns:
        List of article dicts
    """
    scraper = ProFarmerScraper()

    all_articles = []

    for section in scraper.SECTIONS:
        articles = scraper.scrape_section(section, days_back=days_back)
        all_articles.extend(articles)

    print(
        f"[ProFarmer] Total: {len(all_articles)} articles from {len(scraper.SECTIONS)} sections"
    )
    return all_articles


if __name__ == "__main__":
    import json
    import sys

    # Parse command line args
    days_back = 7
    if len(sys.argv) > 1:
        days_back = int(sys.argv[1])

    articles = fetch_profarmer_articles(days_back=days_back)

    # Output JSON for Trigger.dev consumption
    print(json.dumps(articles, indent=2))
