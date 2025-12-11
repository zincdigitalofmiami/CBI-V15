#!/usr/bin/env python3
"""
Tariff Bucket News Collector

Sources:
- Immigration Impact (policy changes)
- SPLC Immigrant Justice (labor/policy)
- Farm Bureau newsrooms (trade policy)
- State ag departments (policy implementation)
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

BUCKET_NAME = "tariff"


def scrape_immigration_impact() -> List[Dict[str, Any]]:
    """Scrape Immigration Impact for policy news"""
    url = "https://immigrationimpact.com/"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        article_items = soup.select('article') or soup.select('div.post')
        
        for item in article_items[:10]:
            title_elem = item.select_one('h2 a') or item.select_one('h3 a')
            
            if not title_elem:
                continue
            
            title = title_elem.get_text(strip=True)
            
            # Filter for agriculture/labor related
            if not any(kw in title.lower() for kw in ['farm', 'agriculture', 'worker', 'labor', 'h-2a', 'visa']):
                continue
            
            article_url = title_elem.get('href', '')
            article_id = hashlib.md5(article_url.encode()).hexdigest()
            
            articles.append({
                "article_id": article_id,
                "headline": title,
                "content": "",
                "source": "Immigration Impact",
                "source_trust_score": 0.80,
                "published_at": datetime.utcnow().isoformat(),
                "url": article_url,
                "bucket_name": BUCKET_NAME
            })
        
        print(f"[{BUCKET_NAME}] Immigration Impact: {len(articles)} articles")
        return articles
    
    except Exception as e:
        print(f"[{BUCKET_NAME}] Error scraping Immigration Impact: {e}")
        return []


def scrape_farm_bureau() -> List[Dict[str, Any]]:
    """Scrape Farm Bureau newsroom"""
    url = "https://www.fb.org/newsroom/"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        article_items = soup.select('div.news-item') or soup.select('article')
        
        for item in article_items[:15]:
            title_elem = item.select_one('h3 a') or item.select_one('h2 a')
            
            if not title_elem:
                continue
            
            title = title_elem.get_text(strip=True)
            
            # Filter for trade/tariff related
            if not any(kw in title.lower() for kw in ['trade', 'tariff', 'export', 'import', 'china', 'ustr']):
                continue
            
            article_url = title_elem.get('href', '')
            if not article_url.startswith('http'):
                article_url = f"https://www.fb.org{article_url}"
            
            article_id = hashlib.md5(article_url.encode()).hexdigest()
            
            articles.append({
                "article_id": article_id,
                "headline": title,
                "content": "",
                "source": "Farm Bureau",
                "source_trust_score": 0.85,
                "published_at": datetime.utcnow().isoformat(),
                "url": article_url,
                "bucket_name": BUCKET_NAME
            })
        
        print(f"[{BUCKET_NAME}] Farm Bureau: {len(articles)} articles")
        return articles
    
    except Exception as e:
        print(f"[{BUCKET_NAME}] Error scraping Farm Bureau: {e}")
        return []


def scrape_state_ag_departments() -> List[Dict[str, Any]]:
    """Scrape state ag department news (CA, TX, FL, GA)"""
    state_sources = [
        {"name": "California Farm Bureau", "url": "https://www.cfbf.com/news/"},
        {"name": "Texas Agriculture", "url": "https://www.texasagriculture.gov/"},
        {"name": "Florida Agriculture", "url": "https://www.fdacs.gov/"},
        {"name": "Georgia Farm Bureau", "url": "https://www.gfb.org/"},
    ]
    
    all_articles = []
    
    for source in state_sources:
        try:
            response = requests.get(source["url"], timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_items = soup.select('article') or soup.select('div.news-item')[:5]
            
            for item in article_items:
                title_elem = item.select_one('h2 a') or item.select_one('h3 a')
                
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                article_url = title_elem.get('href', '')
                
                if not article_url.startswith('http'):
                    base_url = source["url"].rstrip('/')
                    article_url = f"{base_url}{article_url}"
                
                article_id = hashlib.md5(article_url.encode()).hexdigest()
                
                all_articles.append({
                    "article_id": article_id,
                    "headline": title,
                    "content": "",
                    "source": source["name"],
                    "source_trust_score": 0.85,
                    "published_at": datetime.utcnow().isoformat(),
                    "url": article_url,
                    "bucket_name": BUCKET_NAME
                })
            
            print(f"[{BUCKET_NAME}] {source['name']}: {len(article_items)} articles")
        
        except Exception as e:
            print(f"[{BUCKET_NAME}] Error scraping {source['name']}: {e}")
            continue
    
    return all_articles


def fetch_tariff_bucket_news() -> List[Dict[str, Any]]:
    """Fetch all tariff bucket news from all sources"""
    all_articles = []
    
    all_articles.extend(scrape_immigration_impact())
    all_articles.extend(scrape_farm_bureau())
    all_articles.extend(scrape_state_ag_departments())
    
    print(f"[{BUCKET_NAME}] Total: {len(all_articles)} articles")
    return all_articles


if __name__ == "__main__":
    articles = fetch_tariff_bucket_news()
    print(f"\nFetched {len(articles)} tariff bucket articles")

