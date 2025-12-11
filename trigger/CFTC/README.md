# CFTC COT Ingestion

Commitment of Traders (COT) data from Commodity Futures Trading Commission.

## Scripts

- `Scripts/cftc_cot_reports.ts` - Weekly COT positioning data (⚠️ Needs creation)

## Guides

- `Guides/CFTC_COT_INGESTION.md` - Complete ingestion pipeline documentation

## Target Tables

- `raw.cftc_cot_disaggregated` - Commodities positioning
- `raw.cftc_cot_tff` - FX & Treasuries positioning

## Schedule

- Weekly (Friday after report release at 3:30 PM ET)
