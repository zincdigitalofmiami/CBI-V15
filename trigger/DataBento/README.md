# DataBento Ingestion

Market data ingestion from DataBento API.

## Scripts

- `Scripts/databento_ingest_job.ts` - Main ingestion job for 38 futures symbols (ZL, ZS, ZM, CL, HO, etc.)

## Target Tables

- `raw.databento_futures` - OHLCV + tick data

## Schedule

- ZL (Primary): Every hour
- All others: Every 4 hours
