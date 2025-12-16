#!/usr/bin/env python3
"""
Vegas Intel Bucket Collector

Scrapes Vegas-specific sources for demand signals:
- Eater Vegas: Local restaurant developments
- LV Convention & Visitors Authority: Major events & procurement
- Nevada Dept. of Tourism: Tourism arrivals, F&B demand

Why Vegas matters for ZL:
- Restaurant demand proxy for cooking oil consumption
- Convention/tourism activity correlates with F&B demand
- Leading indicator for broader consumer demand trends
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

SOURCES = {
    "eater_vegas": {
        "url": "https://vegas.eater.com/",
        "trust_score": 0.85,
        "keywords": ["restaurant", "opening", "closing", "food", "dining", "chef"],
    },
    "lvcva": {
        "url": "https://www.lvcva.com/news/",
        "trust_score": 0.90,
        "keywords": ["convention", "event", "attendance", "visitor", "tourism"],
    },
    "nevada_tourism": {
        "url": "https://travelnevada.com/news/",
        "trust_score": 0.90,
        "keywords": ["tourism", "arrivals", "visitor", "spending", "food", "beverage"],
    },
}

def scrape_eater_vegas(max_words: int = 500) -> List[Dict[str, Any]]:
    """Scrape Eater Vegas for restaurant news"""
    articles = []
    
    try:
        response = requests.get(SOURCES["eater_vegas"]["url"], timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links
        article_links = soup.find_all('a', class_=['c-entry-box--compact__image-wrapper', 'c-entry-box--compact'])
        
        for link in article_links[:10]:  # Limit to 10 most recent
            try:
                article_url = link.get('href')
                if not article_url or not article_url.startswith('http'):
                    continue
                
                # Scrape article detail
                article_response = requests.get(article_url, timeout=10)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                headline = article_soup.find('h1')
                headline_text = headline.get_text(strip=True) if headline else "No headline"
                
                # Extract article body
                article_body = article_soup.find('div', class_='c-entry-content')
                if article_body:
                    paragraphs = article_body.find_all('p')
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                    
                    # Limit to max_words
                    words = content.split()
                    if len(words) > max_words:
                        content = ' '.join(words[:max_words]) + "..."
                else:
                    content = ""
                
                # Extract date
                date_elem = article_soup.find('time')
                published_at = date_elem.get('datetime') if date_elem else datetime.now().isoformat()
                
                articles.append({
                    "headline": headline_text,
                    "content": content,
                    "url": article_url,
                    "source": "Eater Vegas",
                    "source_trust_score": SOURCES["eater_vegas"]["trust_score"],
                    "published_at": published_at,
                    "bucket_name": "vegas_intel",
                })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error scraping Eater Vegas article: {e}")
                continue
    
    except Exception as e:
        print(f"Error scraping Eater Vegas: {e}")
    
    return articles

def scrape_lvcva(max_words: int = 500) -> List[Dict[str, Any]]:
    """Scrape LV Convention & Visitors Authority"""
    articles = []
    
    try:
        response = requests.get(SOURCES["lvcva"]["url"], timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find news items
        news_items = soup.find_all('article', class_='news-item')
        
        for item in news_items[:10]:
            try:
                headline_elem = item.find('h3') or item.find('h2')
                headline = headline_elem.get_text(strip=True) if headline_elem else "No headline"
                
                link_elem = item.find('a')
                url = link_elem.get('href') if link_elem else ""
                if url and not url.startswith('http'):
                    url = f"https://www.lvcva.com{url}"
                
                summary_elem = item.find('p')
                content = summary_elem.get_text(strip=True) if summary_elem else ""
                
                # Limit to max_words
                words = content.split()
                if len(words) > max_words:
                    content = ' '.join(words[:max_words]) + "..."
                
                articles.append({
                    "headline": headline,
                    "content": content,
                    "url": url,
                    "source": "LVCVA",
                    "source_trust_score": SOURCES["lvcva"]["trust_score"],
                    "published_at": datetime.now().isoformat(),
                    "bucket_name": "vegas_intel",
                })
                
            except Exception as e:
                print(f"Error scraping LVCVA item: {e}")
                continue
    
    except Exception as e:
        print(f"Error scraping LVCVA: {e}")
    
    return articles

def scrape_nevada_tourism(max_words: int = 500) -> List[Dict[str, Any]]:
    """Scrape Nevada Department of Tourism"""
    articles = []
    
    try:
        response = requests.get(SOURCES["nevada_tourism"]["url"], timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find news articles
        news_articles = soup.find_all('article')
        
        for article in news_articles[:10]:
            try:
                headline_elem = article.find('h2') or article.find('h3')
                headline = headline_elem.get_text(strip=True) if headline_elem else "No headline"
                
                link_elem = article.find('a')
                url = link_elem.get('href') if link_elem else ""
                
                content_elem = article.find('p')
                content = content_elem.get_text(strip=True) if content_elem else ""
                
                # Limit to max_words
                words = content.split()
                if len(words) > max_words:
                    content = ' '.join(words[:max_words]) + "..."
                
                articles.append({
                    "headline": headline,
                    "content": content,
                    "url": url,
                    "source": "Nevada Tourism",
                    "source_trust_score": SOURCES["nevada_tourism"]["trust_score"],
                    "published_at": datetime.now().isoformat(),
                    "bucket_name": "vegas_intel",
                })
                
            except Exception as e:
                print(f"Error scraping Nevada Tourism item: {e}")
                continue
    
    except Exception as e:
        print(f"Error scraping Nevada Tourism: {e}")
    
    return articles

def collect_vegas_intel() -> List[Dict[str, Any]]:
    """Collect all Vegas Intel sources"""
    all_articles = []
    
    print("[Vegas Intel] Scraping Eater Vegas...")
    all_articles.extend(scrape_eater_vegas())
    
    print("[Vegas Intel] Scraping LVCVA...")
    all_articles.extend(scrape_lvcva())
    
    print("[Vegas Intel] Scraping Nevada Tourism...")
    all_articles.extend(scrape_nevada_tourism())
    
    print(f"[Vegas Intel] Collected {len(all_articles)} articles")
    return all_articles

if __name__ == "__main__":
    import json
    articles = collect_vegas_intel()
    print(json.dumps(articles, indent=2))

