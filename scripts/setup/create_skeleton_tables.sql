-- Create Skeleton Tables in BigQuery
-- No joins, just table structure with partitioning/clustering
-- Run this after datasets are created

-- ============================================================================
-- RAW LAYER
-- ============================================================================

-- Market Data (Databento)
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.databento_futures_ohlcv_1d` (
  date DATE,
  symbol STRING,
  open FLOAT64,
  high FLOAT64,
  low FLOAT64,
  close FLOAT64,
  volume INT64,
  open_interest INT64
)
PARTITION BY date
CLUSTER BY symbol;

-- FRED Macro Data
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.fred_economic` (
  date DATE,
  series_id STRING,
  value FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY series_id;

-- USDA Reports
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.usda_reports` (
  report_date DATE,
  report_type STRING,
  commodity STRING,
  metric STRING,
  value FLOAT64
)
PARTITION BY DATE(report_date)
CLUSTER BY report_type;

-- CFTC COT Data
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.cftc_cot` (
  date DATE,
  symbol STRING,
  category STRING,
  long_positions INT64,
  short_positions INT64,
  net_positions INT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- EIA Biofuels Data
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.eia_biofuels` (
  date DATE,
  series_id STRING,
  value FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY series_id;

-- Weather Data
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.weather_noaa` (
  date DATE,
  station_id STRING,
  region STRING,
  metric STRING,
  value FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY region;

-- ScrapeCreators Data
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.scrapecreators_trump` (
  date DATE,
  post_id STRING,
  content STRING,
  policy_score FLOAT64
)
PARTITION BY DATE(date);

-- ============================================================================
-- STAGING LAYER
-- ============================================================================

-- Market Daily (Cleaned)
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.market_daily` (
  date DATE,
  symbol STRING,
  open FLOAT64,
  high FLOAT64,
  low FLOAT64,
  close FLOAT64,
  volume INT64,
  open_interest INT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- FRED Macro Clean
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.fred_macro_clean` (
  date DATE,
  series_id STRING,
  value FLOAT64,
  forward_filled BOOL
)
PARTITION BY DATE(date)
CLUSTER BY series_id;

-- USDA Reports Clean
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.usda_reports_clean` (
  report_date DATE,
  report_type STRING,
  commodity STRING,
  metric STRING,
  value FLOAT64
)
PARTITION BY DATE(report_date)
CLUSTER BY report_type;

-- CFTC Positions
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.cftc_positions` (
  date DATE,
  symbol STRING,
  category STRING,
  net_positions INT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- EIA Biofuels Clean
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.eia_biofuels_clean` (
  date DATE,
  series_id STRING,
  value FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY series_id;

-- Weather Aggregated
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.weather_regions_aggregated` (
  date DATE,
  region STRING,
  metric STRING,
  value FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY region;

-- Trump Policy Intelligence
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.trump_policy_intelligence` (
  date DATE,
  event_type STRING,
  zl_impact_score FLOAT64
)
PARTITION BY DATE(date);

-- ============================================================================
-- FEATURES LAYER
-- ============================================================================

-- Technical Indicators
CREATE TABLE IF NOT EXISTS `cbi-v15.features.technical_indicators_us_oil_solutions` (
  date DATE,
  symbol STRING,
  dist_ema_5 FLOAT64,
  dist_ema_10 FLOAT64,
  dist_ema_21 FLOAT64,
  dist_sma_63 FLOAT64,
  dist_sma_200 FLOAT64,
  bb_pct_b FLOAT64,
  bb_bandwidth FLOAT64,
  ppo_12_26 FLOAT64,
  dist_vwap_21d FLOAT64,
  vol_garman_klass_annualized FLOAT64,
  vol_parkinson_annualized FLOAT64,
  vol_21d FLOAT64,
  amihud_illiquidity FLOAT64,
  oi_volume_ratio FLOAT64,
  boho_spread FLOAT64,
  corr_zl_brl_60d FLOAT64,
  terms_of_trade_zl_brl FLOAT64,
  doy_sin FLOAT64,
  doy_cos FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- FX Indicators
CREATE TABLE IF NOT EXISTS `cbi-v15.features.fx_indicators_daily` (
  date DATE,
  currency_pair STRING,
  brl_momentum_21d FLOAT64,
  brl_momentum_63d FLOAT64,
  brl_momentum_252d FLOAT64,
  dxy_momentum_21d FLOAT64,
  dxy_momentum_63d FLOAT64,
  dxy_momentum_252d FLOAT64,
  brl_volatility_21d FLOAT64,
  brl_volatility_63d FLOAT64,
  corr_zl_brl_30d FLOAT64,
  corr_zl_brl_60d FLOAT64,
  corr_zl_brl_90d FLOAT64,
  corr_zl_dxy_30d FLOAT64,
  corr_zl_dxy_60d FLOAT64,
  corr_zl_dxy_90d FLOAT64,
  terms_of_trade_zl_brl FLOAT64,
  corr_regime_zl_brl STRING,
  corr_regime_zl_dxy STRING
)
PARTITION BY DATE(date)
CLUSTER BY currency_pair;

-- Fundamental Spreads
CREATE TABLE IF NOT EXISTS `cbi-v15.features.fundamental_spreads_daily` (
  date DATE,
  board_crush FLOAT64,
  oil_share FLOAT64,
  hog_spread_feeder_margin FLOAT64,
  boho_spread_gal FLOAT64,
  china_pulse_corr_60d FLOAT64
)
PARTITION BY DATE(date);

-- Pair Correlations
CREATE TABLE IF NOT EXISTS `cbi-v15.features.pair_correlations_daily` (
  date DATE,
  symbol_pair STRING,
  symbol1 STRING,
  symbol2 STRING,
  corr_30d FLOAT64,
  corr_60d FLOAT64,
  corr_90d FLOAT64,
  corr_252d FLOAT64,
  corr_regime_60d STRING
)
PARTITION BY DATE(date)
CLUSTER BY symbol_pair;

-- Cross-Asset Betas
CREATE TABLE IF NOT EXISTS `cbi-v15.features.cross_asset_betas_daily` (
  date DATE,
  asset STRING,
  beta_30d FLOAT64,
  beta_60d FLOAT64,
  beta_90d FLOAT64,
  beta_252d FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY asset;

-- Lagged Features
CREATE TABLE IF NOT EXISTS `cbi-v15.features.lagged_features_daily` (
  date DATE,
  symbol STRING,
  price_lag_1d FLOAT64,
  price_lag_2d FLOAT64,
  price_lag_3d FLOAT64,
  price_lag_5d FLOAT64,
  price_lag_10d FLOAT64,
  price_lag_21d FLOAT64,
  return_lag_1d FLOAT64,
  return_lag_2d FLOAT64,
  return_lag_3d FLOAT64,
  return_lag_5d FLOAT64,
  return_lag_10d FLOAT64,
  return_lag_21d FLOAT64,
  price_current FLOAT64,
  return_current FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- Daily ML Matrix (Master Join Table)
CREATE TABLE IF NOT EXISTS `cbi-v15.features.daily_ml_matrix` (
  date DATE,
  symbol STRING,
  -- All 276 features will be joined here
  -- Structure defined in Dataform
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- ============================================================================
-- TRAINING LAYER
-- ============================================================================

-- ZL Training Tables (structure will be populated by Dataform)
CREATE TABLE IF NOT EXISTS `cbi-v15.training.zl_training_1w` (
  date DATE,
  symbol STRING,
  target_zl_1w FLOAT64,
  regime_weight FLOAT64
  -- Features will be added by Dataform
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

CREATE TABLE IF NOT EXISTS `cbi-v15.training.zl_training_1m` (
  date DATE,
  symbol STRING,
  target_zl_1m FLOAT64,
  regime_weight FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

CREATE TABLE IF NOT EXISTS `cbi-v15.training.zl_training_3m` (
  date DATE,
  symbol STRING,
  target_zl_3m FLOAT64,
  regime_weight FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

CREATE TABLE IF NOT EXISTS `cbi-v15.training.zl_training_6m` (
  date DATE,
  symbol STRING,
  target_zl_6m FLOAT64,
  regime_weight FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- ============================================================================
-- FORECASTS LAYER
-- ============================================================================

-- ZL Predictions Tables
CREATE TABLE IF NOT EXISTS `cbi-v15.forecasts.zl_predictions_1w` (
  date DATE,
  model_type STRING,
  prediction FLOAT64,
  target FLOAT64,
  horizon STRING
)
PARTITION BY DATE(date)
CLUSTER BY model_type;

CREATE TABLE IF NOT EXISTS `cbi-v15.forecasts.zl_predictions_1m` (
  date DATE,
  model_type STRING,
  prediction FLOAT64,
  target FLOAT64,
  horizon STRING
)
PARTITION BY DATE(date)
CLUSTER BY model_type;

CREATE TABLE IF NOT EXISTS `cbi-v15.forecasts.zl_predictions_3m` (
  date DATE,
  model_type STRING,
  prediction FLOAT64,
  target FLOAT64,
  horizon STRING
)
PARTITION BY DATE(date)
CLUSTER BY model_type;

CREATE TABLE IF NOT EXISTS `cbi-v15.forecasts.zl_predictions_6m` (
  date DATE,
  model_type STRING,
  prediction FLOAT64,
  target FLOAT64,
  horizon STRING
)
PARTITION BY DATE(date)
CLUSTER BY model_type;

