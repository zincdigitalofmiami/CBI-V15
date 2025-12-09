#!/usr/bin/env python3
"""
China Bucket News Collector

Sources:
- Agrimoney China news
- CONAB Brazil (China's main supplier)
- ABIOVE Brazil oilseed stats
- Reuters commodities (China-related)
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

BUCKET_NAME = "china"


def scrape_agrimoney_china() -> List[Dict[str, Any]]:
    """Scrape Agrimoney China news section"""
    url = "https://www.agrimoney.com/news/china/"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # Find article elements (adjust selectors based on actual HTML)
        article_items = soup.select('article.news-item') or soup.select('div.article')
        
        for item in article_items[:15]:  # Limit to 15 most recent
            title_elem = item.select_one('h2 a') or item.select_one('h3 a') or item.select_one('a')
            snippet_elem = item.select_one('p.excerpt') or item.select_one('div.summary')
            
            if not title_elem:
                continue
            
            article_url = title_elem.get('href', '')
            if not article_url.startswith('http'):
                article_url = f"https://www.agrimoney.com{article_url}"
            
            title = title_elem.get_text(strip=True)
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            
            article_id = hashlib.md5(article_url.encode()).hexdigest()
            
            articles.append({
                "article_id": article_id,
                "headline": title,
                "content": snippet,
                "source": "Agrimoney China",
                "source_trust_score": 0.85,
                "published_at": datetime.utcnow().isoformat(),
                "url": article_url,
                "bucket_name": BUCKET_NAME
            })
        
        print(f"[{BUCKET_NAME}] Agrimoney: {len(articles)} articles")
        return articles
    
    except Exception as e:
        print(f"[{BUCKET_NAME}] Error scraping Agrimoney: {e}")
        return []


def scrape_conab_brazil() -> List[Dict[str, Any]]:
    """Scrape CONAB Brazil news (Portuguese)"""
    url = "https://www.conab.gov.br/ultimas-noticias"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # Find news items
        news_items = soup.select('div.news-item') or soup.select('article')
        
        for item in news_items[:10]:
            title_elem = item.select_one('h3 a') or item.select_one('h2 a')
            
            if not title_elem:
                continue
            
            article_url = title_elem.get('href', '')
            if not article_url.startswith('http'):
                article_url = f"https://www.conab.gov.br{article_url}"
            
            title = title_elem.get_text(strip=True)
            
            # Filter for soybean/oilseed related (Portuguese keywords)
            if not any(kw in title.lower() for kw in ['soja', 'oleaginosas', 'exportação', 'china']):
                continue
            
            article_id = hashlib.md5(article_url.encode()).hexdigest()
            
            articles.append({
                "article_id": article_id,
                "headline": title,
                "content": "",  # Would need to scrape detail page
                "source": "CONAB Brazil",
                "source_trust_score": 0.90,  # Official government source
                "published_at": datetime.utcnow().isoformat(),
                "url": article_url,
                "bucket_name": BUCKET_NAME
            })
        
        print(f"[{BUCKET_NAME}] CONAB: {len(articles)} articles")
        return articles
    
    except Exception as e:
        print(f"[{BUCKET_NAME}] Error scraping CONAB: {e}")
        return []


def scrape_reuters_commodities_china() -> List[Dict[str, Any]]:
    """Scrape Reuters commodities for China-related news"""
    url = "https://www.reuters.com/business/commodities/"
    
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # Reuters uses data-testid attributes
        article_items = soup.select('[data-testid="MediaStoryCard"]') or soup.select('article')
        
        for item in article_items[:20]:
            title_elem = item.select_one('h3 a') or item.select_one('a[data-testid="Heading"]')
            
            if not title_elem:
                continue
            
            title = title_elem.get_text(strip=True)
            
            # Filter for China-related
            if not any(kw in title.lower() for kw in ['china', 'chinese', 'beijing', 'sinograin', 'cofco']):
                continue
            
            article_url = title_elem.get('href', '')
            if not article_url.startswith('http'):
                article_url = f"https://www.reuters.com{article_url}"
            
            article_id = hashlib.md5(article_url.encode()).hexdigest()
            
            articles.append({
                "article_id": article_id,
                "headline": title,
                "content": "",
                "source": "Reuters Commodities",
                "source_trust_score": 0.95,  # Premium news source
                "published_at": datetime.utcnow().isoformat(),
                "url": article_url,
                "bucket_name": BUCKET_NAME
            })
        
        print(f"[{BUCKET_NAME}] Reuters: {len(articles)} China-related articles")
        return articles
    
    except Exception as e:
        print(f"[{BUCKET_NAME}] Error scraping Reuters: {e}")
        return []


def fetch_china_bucket_news() -> List[Dict[str, Any]]:
    """Fetch all China bucket news from all sources"""
    all_articles = []
    
    all_articles.extend(scrape_agrimoney_china())
    all_articles.extend(scrape_conab_brazil())
    all_articles.extend(scrape_reuters_commodities_china())
    
    print(f"[{BUCKET_NAME}] Total: {len(all_articles)} articles from 3 sources")
    return all_articles


if __name__ == "__main__":
    articles = fetch_china_bucket_news()
    print(f"\nFetched {len(articles)} China bucket articles")

