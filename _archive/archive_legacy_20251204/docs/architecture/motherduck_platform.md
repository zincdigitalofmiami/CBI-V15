# MotherDuck Data Platform

**Status:** Production  
**Last Updated:** December 3, 2025

## Overview

MotherDuck serves as the cloud-hosted DuckDB backend for all ZL soybean oil forecasting data and compute.

## Schemas

- `market_data` - ZL prices, related commodities
- `economic_data` - FRED macro indicators (55+ series)
- `supply_demand` - USDA reports, stock levels, production
- `sentiment_data` - News buckets, CFTC positions, policy signals
- `weather_data` - NOAA/INMET/SMN regional data
- `features` - Engineered features (80-120 production)
- `forecasts` - Model outputs, ensemble results
- `events` - Policy triggers, lobbying, known reports

## Connection

**Vercel Integration:**
- Environment variable: `MOTHERDUCK_TOKEN`
- Connection string: `md:usoil_intelligence`

## Extensions

- `anofox_forecast` - Time-series forecasting (31 models)
- `anofox_tabular` - Data quality and anomaly detection
- `anofox_statistics` - Feature engineering

