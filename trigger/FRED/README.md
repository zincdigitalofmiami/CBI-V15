# FRED Ingestion

Economic data ingestion from Federal Reserve Economic Data (FRED) API.

## Scripts

- `Scripts/fred_seed_harvest.ts` - Discovers and ingests FRED series (24 indicators: rates, spreads, financial conditions, economic indicators)

## Target Tables

- `raw.fred_economic` - Single table for all FRED series

## Schedule

- Daily at 1 AM UTC









