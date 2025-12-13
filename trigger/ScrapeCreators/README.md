# ScrapeCreators Ingestion

News and sentiment data from ScrapeCreators API.

## Scripts

- `Scripts/intelligent_news_pipeline.ts` - End-to-end news pipeline orchestrator
- `Scripts/news_to_signals_openai_agent.ts` - AI-powered signal extraction from news

## Guides

- `Guides/NEWS_PIPELINE.md` - News processing architecture

## Target Tables

- `raw.scrapecreators_news_buckets` - Raw news articles
- `features.news_signals` - Processed signals with sentiment, impact, themes

## Schedule

- Every 15 minutes (real-time monitoring)










