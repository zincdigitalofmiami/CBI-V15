# MotherDuck Schema DDL

**Status:** Initial Draft  
**Last Updated:** December 3, 2025

## Execution Order

Run these scripts IN ORDER against MotherDuck:

```bash
# 1. Initialize (extensions + schemas)
00_motherduck_init.sql

# 2. Create tables (in dependency order)
01_raw_schema.sql
02_raw_staging_schema.sql
03_staging_schema.sql
04_features_schema.sql
05_training_schema.sql
06_forecast_schema.sql
07_reference_schema.sql
08_signals_schema.sql
09_ops_schema.sql
```

## What's Created

### Core Tables (Initial Set)
- ✅ `raw.*` - 8 tables (databento, fred, weather, cftc, usda, eia, scrapecreators)
- ✅ `raw_staging.*` - Pattern defined (runtime creation)
- ✅ `staging.*` - 8 cleaned/normalized tables
- ✅ `features.*` - 2 tables (zl_features, feature_importance_log)
- ✅ `training.*` - 3 tables (daily_ml_matrix, model_backtest_zl, zl_model_runs)
- ✅ `forecast.*` - 4 tables (anofox_panel, neural_panel, ensemble_output, ensemble_config)
- ✅ `reference.*` - 4 tables (driver_group, feature_map, model_registry, regime_calendar)
- ✅ `signals.*` - 3 tables (big_eight_live, driver_score_daily, driver_contribution)
- ✅ `ops.*` - 5 tables (data_quality, pipeline_tests, ingestion_log, pipeline_metrics, anomaly_flags)

**Total Created:** ~37 tables across 8 schemas

## What's Missing (TODO)

### Additional Raw Tables Needed
- [ ] `raw.databento_futures_ohlcv_1h` (hourly OHLCV)
- [ ] `raw.databento_futures_ohlcv_1m` (minute OHLCV for MES)
- [ ] `raw.databento_trades` (tick data)
- [ ] `raw.databento_tbbo` (top of book)
- [ ] `raw.databento_mbp_10` (market by price depth-10)
- [ ] `raw.inmet_brazil_weather` (Brazil weather stations)
- [ ] `raw.smn_argentina_weather` (Argentina weather stations)
- [ ] `raw.barchart_palm_oil` (palm oil futures)
- [ ] `raw.world_bank_pink_sheet` (commodity prices)
- [ ] `raw.epa_rin_prices` (if different from EIA)
- [ ] `raw.ice_margin_changes` (ICE margin requirements)
- [ ] `raw.glide_vegas_intel` (restaurant data)

### Additional Staging Tables
- [ ] `staging.mes_intraday_features` (MES E-mini S&P intraday)
- [ ] `staging.palm_oil_daily` (palm futures + spot)
- [ ] `staging.barchart_palm_daily` (Barchart palm data)
- [ ] `staging.volatility_daily` (VIX + realized vol consolidated)
- [ ] `staging.fx_daily` (BRL, CNY, EUR consolidated)
- [ ] `staging.energy_daily` (WTI, HOBO, RB consolidated)
- [ ] `staging.sentiment_daily` (9-layer sentiment system)

### Additional Features Tables
- [ ] `features.fx_indicators_daily` (FX technicals)
- [ ] `features.sentiment_features_daily` (sentiment engineered)
- [ ] `features.weather_production_weighted` (production-weighted weather)
- [ ] `features.crush_margin_daily` (crush economics)

### Additional Training Tables
- [ ] `training.mes_intraday_matrix_1m` (MES 1-minute training)
- [ ] `training.mes_intraday_matrix_5m` (MES 5-minute training)
- [ ] `training.mes_intraday_matrix_15m` (MES 15-minute training)
- [ ] `training.mes_intraday_matrix_30m`, `training.mes_intraday_matrix_1h`, `training.mes_intraday_matrix_4h`
- [ ] `training.zl_training_prod_1w`, `training.zl_training_prod_1m`, `training.zl_training_prod_3m`, etc. (horizon-specific)
- [ ] `training.train_val_test_splits` (split definitions)

### Additional Reference Tables
- [ ] `reference.symbol_universe` (all symbols traded)
- [ ] `reference.tick_sizes` (per symbol)
- [ ] `reference.calendar_holidays` (exchange holidays)
- [ ] `reference.feature_catalog` (full 276 feature definitions)

### Additional Signals Tables
- [ ] `signals.zl_forecast_live` (real-time forecast updates)
- [ ] `signals.crush_oilshare_daily` (crush margin signals)
- [ ] `signals.energy_proxies_daily` (energy-related signals)
- [ ] `signals.hidden_relationship_signals` (cross-asset correlations)

### Additional Ops Tables
- [ ] `ops.model_drift_log` (performance degradation tracking)
- [ ] `ops.feature_drift_log` (feature distribution drift)
- [ ] `ops.schema_change_log` (DDL change tracking)

## Expansion Strategy

**Phase 1 (Current):** Core ZL forecasting tables (37 tables)

**Phase 2:** Add MES intraday tables (12+ tables)

**Phase 3:** Add alternative data sources (palm, vegas intel, etc.)

**Phase 4:** Add monitoring/drift tables

**Total Expected:** ~80-100 tables across 8 schemas

## Notes

- All tables follow column prefix contract (fred_*, databento_*, weather_*, etc.)
- All tables have explicit PRIMARY KEYs
- Monthly partitioning concept maintained (even if not explicit DDL in DuckDB)
- Idempotent MERGE pattern for all raw data ingestion

