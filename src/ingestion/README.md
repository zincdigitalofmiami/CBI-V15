# Ingestion Module

This directory contains all data ingestion pipelines for the V15 system.

## Structure

```text
src/ingestion/
├── scrape_creator/     # ScrapeCreators news ingestion
│   ├── collect.py      # Main orchestrator
│   └── buckets/        # Per-bucket collectors
├── databento/          # Market data ingestion
└── README.md           # This file
```

## Ownership

**AnoFox** owns all ingestion logic. TSci may trigger ingestion jobs but does not modify these scripts.

## Targets

- **Sink 1**: External Drive `/Volumes/Satechi Hub/CBI-V15/data/raw/`
- **Sink 2**: MotherDuck `raw.*` schema
