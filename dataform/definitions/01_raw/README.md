# Raw Data Layer

This directory contains source data declarations for all external data sources.

## Purpose

Raw layer declares external data sources. No transformations here - just declarations that enable Dataform lineage tracking.

## Data Sources

### Market Data
- `databento_daily.sqlx` - Databento daily OHLCV (ZL, ZS, ZM, CL, HO, FCPO)
- `databento_intraday.sqlx` - Databento intraday ticks (1m, 5m, 1h)

### Economic Data
- `fred_macro.sqlx` - FRED economic series (55-60 series)

### Weather Data
- `noaa_weather.sqlx` - NOAA CDO station data
- `noaa_gfs_forecast.sqlx` - NOAA GFS forecasts
- `inmet_brazil.sqlx` - INMET Brazil weather stations
- `argentina_smn.sqlx` - Argentina SMN weather observations
- `google_public_datasets.sqlx` - Google Public Datasets (GSOD, GFS, Normals, GDELT, BLS, FEC)

### Agricultural Data
- `usda_comprehensive.sqlx` - USDA NASS reports
- `usda_fas_esr.sqlx` - USDA FAS Export Sales Reports

### Positioning Data
- `cftc_cot.sqlx` - CFTC Commitments of Traders

### Energy/Biofuels
- `eia_biofuels.sqlx` - EIA biofuels data

### News/Policy/Sentiment
- `scrapecreators_trump.sqlx` - ScrapeCreators Trump posts
- `scrapecreators_buckets.sqlx` - ScrapeCreators news buckets
- `gdelt_events.sqlx` - GDELT event data
- `fec_contributions.sqlx` - FEC political intelligence

### Vegas Intel
- `glide_vegas.sqlx` - Glide API data

## Declaration Format

```sql
config {
  type: "declaration",
  database: "cbi-v15",
  schema: "${dataform.projectConfig.vars.raw_dataset}",
  name: "databento_daily_ohlcv"
}
```

## Usage

Declarations are referenced in staging layer:
```sql
SELECT * FROM ${ref("databento_daily")}
```

---

**Last Updated**: November 28, 2025

