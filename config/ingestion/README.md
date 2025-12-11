# Ingestion Config

## Purpose
Configuration for data source connections - endpoints, rate limits, symbols, credentials references.

## What Belongs Here
- `sources.yaml` - All data source definitions
- API endpoint URLs
- Rate limit settings
- Symbol lists (ZL, ZS, ZC, etc.)

## What Does NOT Belong Here
- Collector code (→ `trigger/<Source>/Scripts/`)
- Operational scripts (→ `scripts/ops/`)
- API keys (→ `.env` or Keychain)

## Naming Convention
`sources.yaml` is the canonical file. Additional source-specific configs: `{source}_config.yaml`
