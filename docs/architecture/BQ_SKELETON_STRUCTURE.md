# BigQuery Skeleton Structure

**Date**: November 28, 2025  
**Status**: ✅ **LOCKED** - Ready for Implementation

---

## 🎯 Structure Principles

### Partitioning Strategy
- ✅ **All tables**: `PARTITION BY DATE(date)`
- ✅ **Clustering**: `CLUSTER BY symbol` (where applicable)
- ✅ **No Joins**: Skeleton structure = table definitions only
- ✅ **Master Join**: `daily_ml_matrix` joins all features

---

## 📊 Dataset Structure

### 1. `raw` Dataset (Source Data)

**Tables**:
- `databento_futures_ohlcv_1d` - Daily OHLCV from Databento
- `fred_economic` - FRED economic indicators (55-60 series)
- `usda_reports` - USDA reports (WASDE, crop progress, exports)
- `cftc_cot` - CFTC Commitments of Traders positions
- `eia_biofuels` - EIA biofuels data (RIN prices, biodiesel production)
- `weather_noaa` - NOAA weather station data
- `scrapecreators_trump` - Trump policy intelligence

**Partitioning**: `PARTITION BY DATE(date)` (or `report_date` for USDA)

---

### 2. `staging` Dataset (Cleaned Data)

**Tables**:
- `market_daily` - Cleaned daily OHLCV with forward-fill
- `fred_macro_clean` - FRED series with forward-fill, daily interpolation
- `usda_reports_clean` - Cleaned USDA reports
- `cftc_positions` - CFTC COT positions by category
- `eia_biofuels_clean` - Cleaned EIA biofuels data
- `weather_regions_aggregated` - Weather data aggregated by region
- `trump_policy_intelligence` - Trump policy events and ZL impact scores

**Partitioning**: `PARTITION BY DATE(date)` (or `report_date` for USDA)
**Clustering**: `CLUSTER BY symbol` (where applicable)

---

### 3. `features` Dataset (Engineered Features)

**Tables**:
- `technical_indicators_us_oil_solutions` - 19 features
- `fx_indicators_daily` - 16 features
- `fundamental_spreads_daily` - 5 features
- `pair_correlations_daily` - 112 features
- `cross_asset_betas_daily` - 28 features
- `lagged_features_daily` - 96 features
- `daily_ml_matrix` - Master join table (276 features)

**Partitioning**: `PARTITION BY DATE(date)`
**Clustering**: `CLUSTER BY symbol` (or `currency_pair`, `symbol_pair`, `asset`)

---

### 4. `training` Dataset (Training-Ready Data)

**Tables**:
- `zl_training_1w` - Training data for 1w horizon with targets
- `zl_training_1m` - Training data for 1m horizon with targets
- `zl_training_3m` - Training data for 3m horizon with targets
- `zl_training_6m` - Training data for 6m horizon with targets

**Partitioning**: `PARTITION BY DATE(date)`
**Clustering**: `CLUSTER BY symbol`

---

### 5. `forecasts` Dataset (Model Predictions)

**Tables**:
- `zl_predictions_1w` - Model predictions for 1w horizon
- `zl_predictions_1m` - Model predictions for 1m horizon
- `zl_predictions_3m` - Model predictions for 3m horizon
- `zl_predictions_6m` - Model predictions for 6m horizon

**Partitioning**: `PARTITION BY DATE(date)`
**Clustering**: `CLUSTER BY model_type`

---

## ✅ Implementation Checklist

### Before Baselines
- ✅ Create all skeleton tables (no joins)
- ✅ Implement USDA ingestion (WASDE, crop progress, exports)
- ✅ Implement CFTC ingestion (COT positions, managed money)
- ✅ Implement EIA ingestion (RIN prices, biodiesel production)
- ✅ Verify partitioning/clustering on all tables

### During Implementation
- ✅ Build feature tables incrementally
- ✅ Test each feature table independently
- ✅ Verify data quality (nulls, outliers)
- ✅ Monitor query costs

### After Implementation
- ✅ Build `daily_ml_matrix` (master join)
- ✅ Export training data
- ✅ Begin baseline training

---

**Last Updated**: November 28, 2025

