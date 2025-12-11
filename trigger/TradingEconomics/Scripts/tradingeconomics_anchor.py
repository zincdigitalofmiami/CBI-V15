#!/usr/bin/env python3
"""
TradingEconomics Soybeans Anchor Scraper

Scrapes news stream from TradingEconomics soybeans commodity page.
Uses Anchor browser automation for JavaScript-rendered content.

URL: https://tradingeconomics.com/commodity/soybeans
Section: News Stream
Edition Type: intraday
"""

import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


def scrape_tradingeconomics_news(max_words_per_article: int = 500) -> List[Dict[str, Any]]:
    """
    Scrape TradingEconomics soybeans news stream.
    
    Args:
        max_words_per_article: Limit article content to first N words
    
    Returns:
        List of article dicts
    """
    url = "https://tradingeconomics.com/commodity/soybeans"
    
    try:
        # Use headers to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # Find news section (adjust selectors based on actual HTML)
        news_section = soup.find('div', {'id': 'news'}) or soup.find('section', class_='news-stream')
        
        if not news_section:
            print("[TradingEconomics] News section not found - may need Anchor for JS rendering")
            return []
        
        # Find news items
        news_items = news_section.find_all('div', class_='news-item') or news_section.find_all('article')
        
        for item in news_items[:20]:  # Limit to 20 most recent
            # Extract timestamp
            time_elem = item.find('time') or item.find('span', class_='date')
            timestamp = datetime.utcnow().isoformat()
            if time_elem and time_elem.has_attr('datetime'):
                timestamp = time_elem['datetime']
            
            # Extract title
            title_elem = item.find('h3') or item.find('h4') or item.find('a', class_='title')
            if not title_elem:
                continue
            
            title = title_elem.get_text(strip=True)
            
            # Extract summary
            summary_elem = item.find('p', class_='summary') or item.find('div', class_='excerpt')
            summary = summary_elem.get_text(strip=True) if summary_elem else ""
            
            # Extract article link
            link_elem = item.find('a', href=True)
            article_url = link_elem['href'] if link_elem else ""
            
            if article_url and not article_url.startswith('http'):
                article_url = f"https://tradingeconomics.com{article_url}"
            
            # Scrape full article if link available
            full_content = summary
            if article_url:
                try:
                    full_content = scrape_article_detail(article_url, max_words_per_article)
                except Exception as e:
                    print(f"[TradingEconomics] Error scraping article detail: {e}")
                    full_content = summary
            
            # Generate article_id
            article_id = hashlib.md5(article_url.encode() if article_url else title.encode()).hexdigest()
            
            articles.append({
                "article_id": article_id,
                "headline": title,
                "content": full_content,
                "source": "TradingEconomics",
                "source_trust_score": 0.85,
                "published_at": timestamp,
                "url": article_url or url,
                "bucket_name": "tradingeconomics_anchor",
                "section": "Soybeans",
                "edition_type": "intraday"
            })
        
        print(f"[TradingEconomics] Scraped {len(articles)} news items")
        return articles
    
    except Exception as e:
        print(f"[TradingEconomics] Error: {e}")
        return []


def scrape_article_detail(url: str, max_words: int = 500) -> str:
    """
    Scrape full article content, limited to first N words.
    
    Args:
        url: Article URL
        max_words: Maximum words to extract
    
    Returns:
        Article content (first N words)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find article body (adjust selectors)
    article_body = soup.find('div', class_='article-body') or soup.find('article') or soup.find('div', class_='content')
    
    if not article_body:
        return ""
    
    # Extract text
    text = article_body.get_text(separator=' ', strip=True)
    
    # Limit to first N words
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words]) + "..."
    
    return text


if __name__ == "__main__":
    articles = scrape_tradingeconomics_news(max_words_per_article=500)
    print(f"\nFetched {len(articles)} TradingEconomics articles")
    
    if articles:
        print("\nSample article:")
        sample = articles[0]
        print(f"  Title: {sample['headline']}")
        print(f"  Content length: {len(sample['content'])} chars")
        print(f"  URL: {sample['url']}")

