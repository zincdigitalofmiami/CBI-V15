#!/usr/bin/env python3
"""
Test ScrapeCreators News Pipeline

Tests:
1. Individual bucket collectors
2. Deduplication logic
3. Metadata enrichment
4. Database schema compatibility
5. End-to-end pipeline
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.scrapecreators.buckets import (
    collect_biofuel_policy,
    collect_china_demand,
    collect_tariffs_trade_policy,
    collect_trump_truth_social,
)


def test_bucket_collector(module, bucket_name):
    """Test a single bucket collector"""
    print(f"\n{'='*60}")
    print(f"Testing {bucket_name} collector...")
    print(f"{'='*60}")
    
    try:
        items = module.fetch_bucket_items()
        print(f"✅ Fetched {len(items)} items")
        
        if items:
            sample = items[0]
            print(f"\nSample item:")
            print(f"  article_id: {sample.get('article_id')}")
            print(f"  headline: {sample.get('headline', '')[:80]}...")
            print(f"  source: {sample.get('source')}")
            print(f"  bucket_name: {sample.get('bucket_name')}")
            print(f"  url: {sample.get('url', 'N/A')[:60]}...")
            print(f"  search_query: {sample.get('search_query', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_deduplication():
    """Test deduplication logic"""
    print(f"\n{'='*60}")
    print(f"Testing deduplication...")
    print(f"{'='*60}")
    
    from src.ingestion.scrapecreators.collect_news_buckets import deduplicate_items
    
    # Create test items with duplicates
    test_items = [
        {"article_id": "1", "url": "http://example.com/a", "source_trust_score": 0.8},
        {"article_id": "2", "url": "http://example.com/b", "source_trust_score": 0.7},
        {"article_id": "3", "url": "http://example.com/a", "source_trust_score": 0.9},  # Duplicate URL, higher trust
        {"article_id": "1", "url": "http://example.com/c", "source_trust_score": 0.6},  # Duplicate ID
    ]
    
    unique = deduplicate_items(test_items)
    
    print(f"Input: {len(test_items)} items")
    print(f"Output: {len(unique)} unique items")
    
    # Should keep item 3 (highest trust for URL a) and item 2
    assert len(unique) == 2, f"Expected 2 unique items, got {len(unique)}"
    
    urls = [item["url"] for item in unique]
    assert "http://example.com/a" in urls
    assert "http://example.com/b" in urls
    
    # Check that we kept the higher trust version
    item_a = [item for item in unique if item["url"] == "http://example.com/a"][0]
    assert item_a["article_id"] == "3", "Should keep higher trust version"
    
    print("✅ Deduplication working correctly")
    return True


def test_metadata_enrichment():
    """Test metadata enrichment"""
    print(f"\n{'='*60}")
    print(f"Testing metadata enrichment...")
    print(f"{'='*60}")
    
    from src.ingestion.scrapecreators.collect_news_buckets import enrich_metadata
    
    # Test HIGH impact
    high_impact_item = {
        "headline": "BREAKING: China bans soybean imports",
        "content": "Emergency measures announced"
    }
    bucket_meta = {"horizon": "TACTICAL"}
    result = enrich_metadata(high_impact_item, bucket_meta)
    
    assert result["impact_magnitude"] == "HIGH", f"Expected HIGH, got {result['impact_magnitude']}"
    assert result["horizon"] == "FLASH", f"Expected FLASH, got {result['horizon']}"
    print("✅ HIGH impact + FLASH horizon detected correctly")
    
    # Test MEDIUM impact
    medium_impact_item = {
        "headline": "USDA releases new soybean forecast",
        "content": "Data shows increase in production"
    }
    result = enrich_metadata(medium_impact_item, bucket_meta)
    
    assert result["impact_magnitude"] == "MEDIUM", f"Expected MEDIUM, got {result['impact_magnitude']}"
    print("✅ MEDIUM impact detected correctly")
    
    # Test STRUCTURAL horizon
    structural_item = {
        "headline": "New EPA regulation on biofuels",
        "content": "Long-term mandate changes announced"
    }
    result = enrich_metadata(structural_item, bucket_meta)
    
    assert result["horizon"] == "STRUCTURAL", f"Expected STRUCTURAL, got {result['horizon']}"
    print("✅ STRUCTURAL horizon detected correctly")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SCRAPECREATORS PIPELINE TEST SUITE")
    print("="*60)
    
    # Check API key
    api_key = os.getenv("SCRAPECREATORS_API_KEY")
    if not api_key:
        print("❌ SCRAPECREATORS_API_KEY not set in environment")
        print("   Set it in .env file or export it")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    results = []
    
    # Test individual collectors
    collectors = [
        (collect_biofuel_policy, "Biofuel Policy"),
        (collect_china_demand, "China Demand"),
        (collect_tariffs_trade_policy, "Tariffs/Trade Policy"),
        (collect_trump_truth_social, "Trump Truth Social"),
    ]
    
    for module, name in collectors:
        results.append(test_bucket_collector(module, name))
    
    # Test deduplication
    results.append(test_deduplication())
    
    # Test metadata enrichment
    results.append(test_metadata_enrichment())
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

