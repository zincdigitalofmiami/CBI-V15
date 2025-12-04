# ScrapeCreator Ingestion

This module handles news ingestion from the ScrapeCreators API into the V15 data pipeline.

## Architecture

```text
ScrapeCreators API
        ↓
    collect.py (orchestrator)
        ↓
    ┌───────────────────────────────────────┐
    │ buckets/                              │
    │   ├── collect_biofuel_policy.py       │
    │   ├── collect_china_demand.py         │
    │   ├── collect_tariffs_trade_policy.py │
    │   └── collect_trump_truth_social.py   │
    └───────────────────────────────────────┘
        ↓
    External Drive (Parquet)
    /Volumes/Satechi Hub/CBI-V15/data/raw/scrape_creator/{bucket}/
        ↓
    MotherDuck (raw.scrapecreators_news_buckets)
```

## Usage

```bash
# Set API key
export SCRAPECREATOR_API_KEY="your-api-key"

# Run ingestion
python collect.py
```

## Policy

**REAL DATA ONLY** - No mock data, ever. If the API key is not set, the script will raise an error.
