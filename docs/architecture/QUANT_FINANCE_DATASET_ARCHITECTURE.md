# Quant Finance Dataset Architecture

**Date**: November 28, 2025  
**Inspired by**: GS Quant, JPM, industry best practices  
**Project**: CBI-V15

---

## Architecture Principles

Following quantitative finance industry standards:

1. **Point-in-Time Discipline**: No lookahead bias, PIT-compliant data
2. **Source Prefixing**: All columns prefixed with source (`databento_`, `fred_`, etc.)
3. **Layer Separation**: Clear boundaries between raw, staging, features, training
4. **Version Control**: All transformations version controlled (Dataform)
5. **Reproducibility**: Every feature traceable to source data

---

## Dataset Architecture (9 Datasets)

### 1. `raw` - Raw Source Data
**Purpose**: Source of truth - never modify  
**Pattern**: Declarations only, no transformations  
**Examples**:
- `databento_daily_ohlcv` - Market data
- `fred_macro` - Economic series
- `noaa_weather` - Weather observations
- `scrapecreators_trump` - News/policy data

**Quant Finance Pattern**: Immutable source layer

---

### 2. `staging` - Cleaned Normalized Data
**Purpose**: PIT-compliant cleaned data  
**Pattern**: Forward-filled with limits, deduplicated  
**Examples**:
- `market_daily` - Cleaned OHLCV
- `fred_macro_clean` - Forward-filled economic data
- `weather_regions_aggregated` - Regional aggregates

**Quant Finance Pattern**: Point-in-time discipline enforced

---

### 3. `features` - Engineered Features
**Purpose**: Feature store - versioned, reproducible  
**Pattern**: All feature engineering logic in Dataform  
**Examples**:
- `big_eight_signals` - Big 8 drivers
- `technical_indicators` - RSI, MACD, Bollinger
- `cross_asset_correlations` - Rolling correlations
- `palm_features_daily` - Palm oil features

**Quant Finance Pattern**: Feature store with lineage

---

### 4. `training` - Training-Ready Tables
**Purpose**: Walk-forward validation ready  
**Pattern**: Targets, horizons, regime weights  
**Examples**:
- `zl_training_1w` - 1 week horizon
- `zl_training_1m` - 1 month horizon
- `zl_training_3m` - 3 month horizon
- `zl_training_6m` - 6 month horizon

**Quant Finance Pattern**: Train/val/test splits, walk-forward CV

---

### 5. `forecasts` - Model Predictions
**Purpose**: Prediction store - versioned models  
**Pattern**: Multi-horizon forecasts with uncertainty  
**Examples**:
- `zl_forecasts_daily` - Daily forecasts by horizon
- `zl_forecast_intervals` - Prediction intervals
- `model_metadata` - Model versions, hyperparameters

**Quant Finance Pattern**: Versioned predictions with metadata

---

### 6. `signals` - Trading Signals
**Purpose**: Signal generation layer  
**Pattern**: Derived indicators, trading rules  
**Examples**:
- `zl_entry_signals` - Entry signals
- `zl_exit_signals` - Exit signals
- `regime_signals` - Regime classifications

**Quant Finance Pattern**: Signal generation separate from predictions

---

### 7. `reference` - Reference Data
**Purpose**: Dimension tables - slow-changing  
**Pattern**: Calendars, symbols, mappings  
**Examples**:
- `trading_calendar` - Trading days
- `symbol_mappings` - Symbol metadata
- `regime_calendar` - Regime definitions

**Quant Finance Pattern**: Dimension tables (star schema)

---

### 8. `api` - Public API Views
**Purpose**: Read-only views for consumption  
**Pattern**: Dashboard-ready, optimized queries  
**Examples**:
- `vw_latest_forecast` - Latest forecasts
- `vw_big_eight_signals` - Current Big 8 values
- `vw_regime_status` - Current regime

**Quant Finance Pattern**: Consumption layer (read-only)

---

### 9. `ops` - Operations Monitoring
**Purpose**: Observability layer  
**Pattern**: Data quality, model performance  
**Examples**:
- `data_quality_checks` - Daily quality scores
- `model_performance` - MAE, RMSE, R² by horizon
- `ingestion_logs` - Data ingestion logs

**Quant Finance Pattern**: Observability and monitoring

---

## Comparison with GS Quant / JPM Patterns

| Our Dataset | GS Quant Equivalent | JPM Equivalent |
|------------|---------------------|----------------|
| `raw` | Source data | Raw layer |
| `staging` | Cleaned data | Staging layer |
| `features` | Feature store | Feature engineering |
| `training` | Training data | Model inputs |
| `forecasts` | Predictions | Model outputs |
| `signals` | Signals | Trading signals |
| `reference` | Reference data | Dimension tables |
| `api` | API layer | Consumption layer |
| `ops` | Monitoring | Observability |

---

## Key Differences from Generic Data Warehouses

1. **Point-in-Time Discipline**: No lookahead bias
2. **Regime Awareness**: Regime weights for training
3. **Multi-Horizon**: Separate tables per horizon
4. **Signal Layer**: Separate signals from predictions
5. **Feature Store**: Versioned, reproducible features

---

## APIs Required

### Core APIs (Critical)
- `bigquery.googleapis.com` - Data warehouse
- `dataform.googleapis.com` - ETL framework
- `secretmanager.googleapis.com` - API keys
- `cloudscheduler.googleapis.com` - Daily jobs

### Supporting APIs
- `bigqueryconnection.googleapis.com` - Federated queries
- `bigquerymigration.googleapis.com` - Migration tools
- `cloudfunctions.googleapis.com` - Serverless ingestion
- `run.googleapis.com` - Containerized jobs
- `logging.googleapis.com` - Monitoring
- `monitoring.googleapis.com` - Metrics
- `pubsub.googleapis.com` - Event-driven (optional)

---

## Verification

Run verification script:
```bash
./scripts/setup/verify_apis_and_datasets.sh
```

---

**Last Updated**: November 28, 2025

