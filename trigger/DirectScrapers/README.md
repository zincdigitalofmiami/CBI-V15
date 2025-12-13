# Direct Web Scrapers

Direct website scraping for news buckets (not using ScrapeCreators API).

## Scripts

- `Scripts/collect_china_news.py` - China bucket (Agrimoney, CONAB, Reuters)
- `Scripts/collect_tariff_news.py` - Tariff bucket (Immigration Impact, Farm Bureau, State Ag)

## Target Tables

- `raw.scrapecreators_news_buckets` - News articles by bucket

## Schedule

- Hourly

## Notes

These scrapers fetch directly from source websites. For API-based news collection, see `ScrapeCreators/`.

