#!/usr/bin/env python3
"""
Direct URL Scraper for Premium Sources (DTN, ProFarmer, etc.)

Uses authenticated sessions to scrape paywalled content.
Credentials stored in .env or macOS Keychain.
"""

import os
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
from bs4 import BeautifulSoup


class DirectURLScraper:
    """Scraper for authenticated/paywalled sources"""

    def __init__(self):
        # Load credentials from env
        self.dtn_username = os.getenv("DTN_USERNAME")
        self.dtn_password = os.getenv("DTN_PASSWORD")
        self.jacobsen_username = os.getenv("JACOBSEN_USERNAME")
        self.jacobsen_password = os.getenv("JACOBSEN_PASSWORD")
        self.profarmer_username = os.getenv("PROFARMER_USERNAME")
        self.profarmer_password = os.getenv("PROFARMER_PASSWORD")

        # Session management
        self.sessions = {}

    def _get_session(self, source: str) -> requests.Session:
        """Get or create authenticated session for source"""
        if source in self.sessions:
            return self.sessions[source]

        session = requests.Session()

        if source == "dtn":
            self._login_dtn(session)
        elif source == "jacobsen":
            self._login_jacobsen(session)
        elif source == "profarmer":
            self._login_profarmer(session)

        self.sessions[source] = session
        return session

    def _login_dtn(self, session: requests.Session):
        """Login to DTN/Progressive Farmer"""
        if not self.dtn_username or not self.dtn_password:
            raise RuntimeError("DTN credentials not set")

        login_url = "https://www.dtnpf.com/agriculture/web/ag/login"
        payload = {"username": self.dtn_username, "password": self.dtn_password}

        response = session.post(login_url, data=payload, timeout=30)
        response.raise_for_status()

    def _login_jacobsen(self, session: requests.Session):
        """Login to The Jacobsen"""
        if not self.jacobsen_username or not self.jacobsen_password:
            raise RuntimeError("Jacobsen credentials not set")

        login_url = "https://thejacobsen.com/login"
        payload = {"email": self.jacobsen_username, "password": self.jacobsen_password}

        response = session.post(login_url, data=payload, timeout=30)
        response.raise_for_status()

    def _login_profarmer(self, session: requests.Session):
        """Login to ProFarmer"""
        if not self.profarmer_username or not self.profarmer_password:
            raise RuntimeError("ProFarmer credentials not set")

        login_url = "https://www.profarmer.com/login"
        payload = {
            "username": self.profarmer_username,
            "password": self.profarmer_password,
        }

        response = session.post(login_url, data=payload, timeout=30)
        response.raise_for_status()

    def scrape_url(self, url: str, source: str) -> Dict[str, Any]:
        """
        Scrape a specific URL with authentication.

        Args:
            url: Full URL to scrape
            source: Source identifier (dtn, jacobsen, profarmer)

        Returns:
            Dict with article_id, headline, content, source, published_at
        """
        session = self._get_session(source)

        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Extract content (source-specific selectors)
        if source == "dtn":
            headline = soup.select_one("h1.article-title")
            content_div = soup.select_one("div.article-body")
            date_elem = soup.select_one("time.article-date")
        elif source == "jacobsen":
            headline = soup.select_one("h1.entry-title")
            content_div = soup.select_one("div.entry-content")
            date_elem = soup.select_one("time.entry-date")
        elif source == "profarmer":
            headline = soup.select_one("h1.post-title")
            content_div = soup.select_one("div.post-content")
            date_elem = soup.select_one("span.post-date")
        else:
            # Generic fallback
            headline = soup.select_one("h1")
            content_div = soup.select_one("article") or soup.select_one("main")
            date_elem = soup.select_one("time")

        # Generate article ID from URL
        article_id = hashlib.md5(url.encode()).hexdigest()

        return {
            "article_id": article_id,
            "headline": headline.get_text(strip=True) if headline else "",
            "content": content_div.get_text(strip=True) if content_div else "",
            "source": source,
            "published_at": (
                date_elem.get("datetime")
                if date_elem and date_elem.has_attr("datetime")
                else datetime.utcnow().isoformat()
            ),
            "source_trust_score": 0.95,  # High trust for premium sources
            "url": url,
        }


def scrape_dtn_daily() -> List[Dict[str, Any]]:
    """Scrape DTN daily market updates"""
    scraper = DirectURLScraper()

    # DTN daily URLs (update these with actual daily report URLs)
    urls = [
        "https://www.dtnpf.com/agriculture/web/ag/news/article/2024/12/09/soybean-oil-market-update",
        "https://www.dtnpf.com/agriculture/web/ag/crops/article/2024/12/09/daily-grain-market-summary",
    ]

    items = []
    for url in urls:
        try:
            item = scraper.scrape_url(url, "dtn")
            items.append(item)
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return items


def scrape_jacobsen_daily() -> List[Dict[str, Any]]:
    """Scrape Jacobsen daily RIN/biofuel reports"""
    scraper = DirectURLScraper()

    # Jacobsen daily URLs
    urls = [
        "https://thejacobsen.com/article/daily-rin-prices",
        "https://thejacobsen.com/article/biodiesel-market-update",
    ]

    items = []
    for url in urls:
        try:
            item = scraper.scrape_url(url, "jacobsen")
            items.append(item)
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return items


if __name__ == "__main__":
    print("Testing DTN scraper...")
    dtn_items = scrape_dtn_daily()
    print(f"Scraped {len(dtn_items)} DTN items")

    print("\nTesting Jacobsen scraper...")
    jacobsen_items = scrape_jacobsen_daily()
    print(f"Scraped {len(jacobsen_items)} Jacobsen items")
