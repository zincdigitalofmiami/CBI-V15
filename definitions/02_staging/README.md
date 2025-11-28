# Staging Layer

This directory contains cleaned, normalized, and forward-filled data transformations.

## Purpose

Staging layer transforms raw data into clean, normalized format ready for feature engineering.

## Key Transformations

- **Cleaning**: Remove nulls, handle missing values
- **Normalization**: Standardize date formats, column names
- **Forward Fill**: Fill sparse data with limits (max 30 days)
- **Deduplication**: Remove duplicates
- **Type Conversion**: Ensure correct data types

## Tables

### Market Data
- `market_daily.sqlx` - Cleaned daily OHLCV
- `market_intraday.sqlx` - Aggregated intraday data

### Economic Data
- `fred_macro_clean.sqlx` - FRED series with forward-fill

### Weather Data
- `weather_noaa_daily.sqlx` - NOAA daily aggregates
- `weather_noaa_gfs_forecast.sqlx` - GFS forecast aggregates
- `weather_brazil_inmet.sqlx` - Brazil INMET aggregates
- `weather_argentina_smn.sqlx` - Argentina SMN aggregates
- `weather_regions_aggregated.sqlx` - Regional weighted aggregates
- `weather_buckets.sqlx` - Weather feature buckets (precip, temp, GDD, ENSO)

### Agricultural Data
- `usda_reports_clean.sqlx` - Cleaned WASDE, crop progress
- `usda_export_sales.sqlx` - Export sales by country
- `usda_crop_progress.sqlx` - Crop progress data
- `usda_wasde.sqlx` - WASDE projections

### Positioning Data
- `cftc_positions.sqlx` - COT positions
- `cftc_managed_money.sqlx` - Managed money positions

### Energy/Biofuels
- `eia_biofuels_clean.sqlx` - Cleaned biofuels data
- `eia_rin_prices.sqlx` - RIN prices
- `eia_biodiesel_production.sqlx` - Biodiesel production

### News/Sentiment
- `news_bucketed.sqlx` - All ScrapeCreators buckets aggregated
- `sentiment_buckets.sqlx` - Sentiment scores by bucket
- `trump_policy_intelligence.sqlx` - Trump policy data
- `gdelt_events_clean.sqlx` - GDELT events cleaned
- `fec_ag_energy_pacs.sqlx` - FEC contributions cleaned

### Vegas Intel
- `vegas_restaurants.sqlx` - Cleaned restaurant data

## Incremental Pattern

All staging tables use incremental MERGE:

```sql
config {
  type: "incremental",
  uniqueKey: ["date", "symbol"],
  bigquery: {
    partitionBy: "DATE(date)",
    clusterBy: ["symbol"]
  }
}
```

## Forward Fill Logic

Sparse data (e.g., monthly FRED) forward-filled with limits:

```sql
LAST_VALUE(value IGNORE NULLS) OVER (
  ORDER BY date
  ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
) AS value_ff_30d
```

---

**Last Updated**: November 28, 2025

