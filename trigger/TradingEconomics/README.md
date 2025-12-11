# TradingEconomics Ingestion

Global commodity and economic data from TradingEconomics API (paid).

## Scripts

- `Scripts/tradingeconomics_goldmine.ts` - Comprehensive commodity data ingestion

## Target Tables

- `raw.tradingeconomics` - Global commodity prices, economic calendars, central bank rates

## Schedule

- Daily at 1 AM UTC

## Authentication

Requires `TRADINGECONOMICS_API_KEY` environment variable.
