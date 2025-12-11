# ProFarmer Ingestion

Premium agricultural market intelligence from ProFarmer (paid subscription).

## Scripts

- `Scripts/profarmer_all_urls.ts` - PRIMARY job scraping 22+ URLs (daily editions, news, newsletters, analysis, commodities, weather)
- `Scripts/profarmer_anchor_scraper.ts` - Anchor browser automation helper
- `Scripts/profarmer_ingest_job.ts` - Legacy ingestion job (may be deprecated)

## Target Tables

- `raw.bucket_news` - All ProFarmer articles with bucket tags

## Schedule

- 3x daily (6 AM, 12 PM, 6 PM UTC)

## Authentication

Requires `PROFARMER_USERNAME` and `PROFARMER_PASSWORD` environment variables.
