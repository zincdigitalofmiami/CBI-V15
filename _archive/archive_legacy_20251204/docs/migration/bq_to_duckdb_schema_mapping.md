# BigQuery to DuckDB Schema Mapping

**Date:** December 3, 2024  
**Source:** `scripts/migration/bq_schemas_detailed.json`

---

## Type Mappings

| BigQuery Type | DuckDB Type | Notes |
|---------------|-------------|-------|
| `DATE` | `DATE` | Direct mapping |
| `STRING` | `VARCHAR` | Direct mapping |
| `INT64` | `BIGINT` | Direct mapping |
| `FLOAT64` | `DOUBLE` | Direct mapping |
| `BOOL` | `BOOLEAN` | Direct mapping |
| `TIMESTAMP` | `TIMESTAMP` | Direct mapping |

---

## Dataset Mappings

### Raw Dataset (11 tables → DuckDB `raw` schema)

| BigQuery Table | DuckDB Table | Columns | Key Columns |
|----------------|--------------|---------|-------------|
| `databento_futures_ohlcv_1d` | `raw.databento_futures_ohlcv_1d` | 8 | date, symbol |
| `databento_futures_ohlcv_1h` | `raw.databento_futures_ohlcv_1h` | 9 | date, hour, symbol |
| `databento_futures_statistics` | `raw.databento_futures_statistics` | 8 | date, symbol |
| `fred_economic` | `raw.fred_economic` | 3 | date, series_id |
| `eia_biofuels` | `raw.eia_biofuels` | 3 | date, series_id |
| `cftc_cot` | `raw.cftc_cot` | 6 | date, symbol |
| `usda_reports` | `raw.usda_reports` | 5 | report_date |
| `weather_noaa` | `raw.weather_noaa` | 5 | date, station_id |
| `scrapecreators_trump` | `raw.scrapecreators_trump` | 4 | date, post_id |
| `scrapecreators_news_buckets` | `raw.scrapecreators_news_buckets` | 15 | date, article_id |
| `test_table` | `raw.test_table` | 2 | date |

**Indexes to Create:**
- All tables: `idx_{table}_date` on `date`
- Symbol tables: `idx_{table}_symbol` on `symbol`
- Series tables: `idx_{table}_series_id` on `series_id`

### Staging Dataset (9 tables → DuckDB `staging` schema)

| BigQuery Table | DuckDB Table | Columns | Key Columns |
|----------------|--------------|---------|-------------|
| `market_daily` | `staging.market_daily` | 8 | date, symbol |
| `fred_macro_clean` | `staging.fred_macro_clean` | 4 | date, series_id |
| `news_bucketed` | `staging.news_bucketed` | 11 | date, theme |
| `sentiment_buckets` | `staging.sentiment_buckets` | 8 | date, theme |
| `cftc_positions` | `staging.cftc_positions` | 4 | date, symbol |
| `usda_reports_clean` | `staging.usda_reports_clean` | 5 | report_date |
| `eia_biofuels_clean` | `staging.eia_biofuels_clean` | 3 | date, series_id |
| `weather_regions_aggregated` | `staging.weather_regions_aggregated` | 4 | date, region |
| `trump_policy_intelligence` | `staging.trump_policy_intelligence` | 3 | date |

**Indexes to Create:**
- All tables: `idx_{table}_date` on `date`
- Symbol tables: `idx_{table}_symbol` on `symbol`

### Features Dataset (12 tables → DuckDB `features` schema)

| BigQuery Table | DuckDB Table | Columns | Key Columns |
|----------------|--------------|---------|-------------|
| `daily_ml_matrix` | `features.daily_ml_matrix` | 2 | date, symbol |
| `technical_indicators_us_oil_solutions` | `features.technical_indicators_us_oil_solutions` | 21 | date, symbol |
| `fx_indicators_daily` | `features.fx_indicators_daily` | 19 | date |
| `fundamental_spreads_daily` | `features.fundamental_spreads_daily` | 6 | date |
| `pair_correlations_daily` | `features.pair_correlations_daily` | 9 | date |
| `cross_asset_betas_daily` | `features.cross_asset_betas_daily` | 6 | date, asset |
| `lagged_features_daily` | `features.lagged_features_daily` | 16 | date, symbol |
| `sentiment_features_daily` | `features.sentiment_features_daily` | 9 | date |
| `trump_news_features_daily` | `features.trump_news_features_daily` | 7 | date |
| `regime_indicators_daily` | `features.regime_indicators_daily` | 5 | date |
| `neural_signals_daily` | `features.neural_signals_daily` | 4 | date |
| `neural_master_score` | `features.neural_master_score` | 2 | date |

**Note:** `features.daily_ml_matrix` appears to be a placeholder view. The actual ML matrix is in `training.daily_ml_matrix` (67 columns).

### Training Dataset (5 tables → DuckDB `training` schema)

| BigQuery Table | DuckDB Table | Columns | Key Columns |
|----------------|--------------|---------|-------------|
| `daily_ml_matrix` | `training.daily_ml_matrix` | 67 | date, symbol |
| `zl_training_1w` | `training.zl_training_1w` | 4 | date, symbol |
| `zl_training_1m` | `training.zl_training_1m` | 4 | date, symbol |
| `zl_training_3m` | `training.zl_training_3m` | 4 | date, symbol |
| `zl_training_6m` | `training.zl_training_6m` | 4 | date, symbol |

**Critical:** `training.daily_ml_matrix` is the canonical ML feature matrix with all 276 engineered features.

### Forecasts Dataset (4 tables → DuckDB `forecasts` schema)

| BigQuery Table | DuckDB Table | Columns | Key Columns |
|----------------|--------------|---------|-------------|
| `zl_predictions_1w` | `forecasts.zl_predictions_1w` | 5 | date, symbol |
| `zl_predictions_1m` | `forecasts.zl_predictions_1m` | 5 | date, symbol |
| `zl_predictions_3m` | `forecasts.zl_predictions_3m` | 5 | date, symbol |
| `zl_predictions_6m` | `forecasts.zl_predictions_6m` | 5 | date, symbol |

**Schema:** `date`, `symbol`, `forecast`, `lower_bound`, `upper_bound`

### Reference Dataset (4 tables → DuckDB `reference` schema)

| BigQuery Table | DuckDB Table | Columns | Key Columns |
|----------------|--------------|---------|-------------|
| `regime_calendar` | `reference.regime_calendar` | 6 | start_date, end_date |
| `regime_weights` | `reference.regime_weights` | 7 | regime_type |
| `train_val_test_splits` | `reference.train_val_test_splits` | 3 | split_type |
| `neural_drivers` | `reference.neural_drivers` | 5 | driver_name |

### Ops Dataset (1 table → DuckDB `ops` schema)

| BigQuery Table | DuckDB Table | Columns | Key Columns |
|----------------|--------------|---------|-------------|
| `ingestion_completion` | `ops.ingestion_completion` | 6 | date, source |

---

## Migration Checklist

- [x] Extract BigQuery schemas (`scripts/migration/get_bq_schemas.py`)
- [x] Create DuckDB schemas matching BigQuery structure
- [x] Map data types (DATE → DATE, STRING → VARCHAR, etc.)
- [x] Create indexes on key columns (date, symbol, series_id)
- [x] Verify column counts match
- [ ] Test queries on migrated tables
- [ ] Validate data integrity (row counts, date ranges)

---

**Last Updated:** December 3, 2024

