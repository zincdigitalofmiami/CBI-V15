-- Create Complete Skeleton Tables in BigQuery (42 tables)
-- No joins, just table structure with partitioning/clustering
-- Run this after datasets are created

-- ============================================================================
-- RAW LAYER (8 tables)
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
PARTITION BY DATE(date)
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

-- ScrapeCreators Trump Data
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.scrapecreators_trump` (
  date DATE,
  post_id STRING,
  content STRING,
  policy_score FLOAT64
)
PARTITION BY DATE(date);

-- ScrapeCreators News Buckets (3-Way Segmentation)
CREATE TABLE IF NOT EXISTS `cbi-v15.raw.scrapecreators_news_buckets` (
  date DATE,
  article_id STRING,
  theme_primary STRING,
  is_trump_related BOOL,
  policy_axis STRING,
  horizon STRING,
  zl_sentiment STRING,
  impact_magnitude STRING,
  sentiment_confidence FLOAT64,
  sentiment_raw_score FLOAT64,
  headline STRING,
  content STRING,
  source STRING,
  source_trust_score FLOAT64,
  created_at TIMESTAMP
)
PARTITION BY DATE(date)
CLUSTER BY theme_primary, is_trump_related;

-- ============================================================================
-- STAGING LAYER (9 tables)
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
  value FLOAT64
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
  long_positions INT64,
  short_positions INT64,
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
  post_id STRING,
  content STRING,
  policy_score FLOAT64,
  zl_impact_score FLOAT64
)
PARTITION BY DATE(date);

-- News Bucketed (Aggregated by date, bucket)
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.news_bucketed` (
  date DATE,
  bucket_type STRING,
  article_count INT64,
  avg_confidence FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY bucket_type;

-- Sentiment Buckets (Sentiment scores by bucket)
CREATE TABLE IF NOT EXISTS `cbi-v15.staging.sentiment_buckets` (
  date DATE,
  bucket_type STRING,
  bullish_count INT64,
  bearish_count INT64,
  neutral_count INT64,
  total_count INT64,
  net_sentiment INT64,
  bullish_weighted_score FLOAT64,
  bearish_weighted_score FLOAT64,
  net_weighted_score FLOAT64,
  avg_confidence FLOAT64,
  normalized_sentiment FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY bucket_type;

-- ============================================================================
-- FEATURES LAYER (12 tables)
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
  lag_1d_price FLOAT64,
  lag_2d_price FLOAT64,
  lag_3d_price FLOAT64,
  lag_5d_price FLOAT64,
  lag_10d_price FLOAT64,
  lag_21d_price FLOAT64,
  lag_1d_return FLOAT64,
  lag_2d_return FLOAT64,
  lag_3d_return FLOAT64,
  lag_5d_return FLOAT64,
  lag_10d_return FLOAT64,
  lag_21d_return FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- Daily ML Matrix (Master Join Table)
CREATE TABLE IF NOT EXISTS `cbi-v15.features.daily_ml_matrix` (
  date DATE,
  symbol STRING,
  target_1w_price FLOAT64,
  target_1m_price FLOAT64,
  target_3m_price FLOAT64,
  target_6m_price FLOAT64
  -- Note: All 276 features will be added via Dataform joins
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- Sentiment Features Daily
CREATE TABLE IF NOT EXISTS `cbi-v15.features.sentiment_features_daily` (
  date DATE,
  news_supply_tact_net_7d FLOAT64,
  news_biofuel_tact_net_7d FLOAT64,
  news_trade_struct_net_30d FLOAT64,
  news_macro_risk_net_7d FLOAT64,
  news_logistics_tact_net_7d FLOAT64,
  news_zl_pulse_7d STRING,
  news_sentiment_change_1d FLOAT64,
  news_sentiment_velocity_7d FLOAT64
)
PARTITION BY DATE(date);

-- Regime Indicators Daily
CREATE TABLE IF NOT EXISTS `cbi-v15.features.regime_indicators_daily` (
  date DATE,
  volatility_regime STRING,
  correlation_regime STRING,
  trend_regime STRING,
  regime_weight FLOAT64
)
PARTITION BY DATE(date);

-- Neural Signals Daily (Layer 2)
CREATE TABLE IF NOT EXISTS `cbi-v15.features.neural_signals_daily` (
  date DATE,
  dollar_neural_score FLOAT64,
  fed_neural_score FLOAT64,
  crush_neural_score FLOAT64
)
PARTITION BY DATE(date);

-- Neural Master Score (Layer 1)
CREATE TABLE IF NOT EXISTS `cbi-v15.features.neural_master_score` (
  date DATE,
  neural_master_score FLOAT64
)
PARTITION BY DATE(date);

-- Trump News Features Daily
CREATE TABLE IF NOT EXISTS `cbi-v15.features.trump_news_features_daily` (
  date DATE,
  policy_trump_trade_china_net_7d FLOAT64,
  policy_trump_trade_china_net_30d FLOAT64,
  policy_trump_biofuels_net_7d FLOAT64,
  policy_trump_biofuels_net_30d FLOAT64,
  policy_trump_zl_net_7d FLOAT64,
  policy_trump_zl_net_30d FLOAT64
)
PARTITION BY DATE(date);

-- ============================================================================
-- REFERENCE LAYER (4 tables)
-- ============================================================================

-- Regime Calendar
CREATE TABLE IF NOT EXISTS `cbi-v15.reference.regime_calendar` (
  regime_type STRING,
  start_date DATE,
  end_date DATE,
  description STRING,
  base_weight FLOAT64,
  vix_multiplier FLOAT64
)
CLUSTER BY regime_type;

-- Regime Weights
CREATE TABLE IF NOT EXISTS `cbi-v15.reference.regime_weights` (
  regime_type STRING,
  base_weight FLOAT64,
  vix_multiplier FLOAT64,
  shock_multiplier_policy FLOAT64,
  shock_multiplier_vol FLOAT64,
  shock_multiplier_supply FLOAT64,
  shock_multiplier_geopol FLOAT64
)
CLUSTER BY regime_type;

-- Neural Drivers
CREATE TABLE IF NOT EXISTS `cbi-v15.reference.neural_drivers` (
  layer INT64,
  driver_name STRING,
  description STRING,
  input_features ARRAY<STRING>,
  output_feature STRING
)
CLUSTER BY layer, driver_name;

-- Train/Val/Test Splits
CREATE TABLE IF NOT EXISTS `cbi-v15.reference.train_val_test_splits` (
  set_name STRING,
  start_date DATE,
  end_date DATE,
  description STRING
);

-- ============================================================================
-- OPS LAYER (1 table)
-- ============================================================================

-- Ingestion Completion Tracking
CREATE TABLE IF NOT EXISTS `cbi-v15.ops.ingestion_completion` (
  date DATE,
  source STRING,
  completed_at TIMESTAMP,
  status STRING,
  rows_ingested INT64,
  error_message STRING
)
PARTITION BY DATE(date)
CLUSTER BY source;

-- ============================================================================
-- TRAINING LAYER (4 tables)
-- ============================================================================

-- ZL Training 1w
CREATE TABLE IF NOT EXISTS `cbi-v15.training.zl_training_1w` (
  date DATE,
  symbol STRING,
  target_1w_price FLOAT64,
  regime_weight FLOAT64
  -- Note: All 276 features will be added via Dataform joins
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- ZL Training 1m
CREATE TABLE IF NOT EXISTS `cbi-v15.training.zl_training_1m` (
  date DATE,
  symbol STRING,
  target_1m_price FLOAT64,
  regime_weight FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- ZL Training 3m
CREATE TABLE IF NOT EXISTS `cbi-v15.training.zl_training_3m` (
  date DATE,
  symbol STRING,
  target_3m_price FLOAT64,
  regime_weight FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- ZL Training 6m
CREATE TABLE IF NOT EXISTS `cbi-v15.training.zl_training_6m` (
  date DATE,
  symbol STRING,
  target_6m_price FLOAT64,
  regime_weight FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY symbol;

-- ============================================================================
-- FORECASTS LAYER (4 tables)
-- ============================================================================

-- ZL Predictions 1w
CREATE TABLE IF NOT EXISTS `cbi-v15.forecasts.zl_predictions_1w` (
  date DATE,
  model_type STRING,
  prediction FLOAT64,
  lower_bound FLOAT64,
  upper_bound FLOAT64,
  confidence FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY model_type;

-- ZL Predictions 1m
CREATE TABLE IF NOT EXISTS `cbi-v15.forecasts.zl_predictions_1m` (
  date DATE,
  model_type STRING,
  prediction FLOAT64,
  lower_bound FLOAT64,
  upper_bound FLOAT64,
  confidence FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY model_type;

-- ZL Predictions 3m
CREATE TABLE IF NOT EXISTS `cbi-v15.forecasts.zl_predictions_3m` (
  date DATE,
  model_type STRING,
  prediction FLOAT64,
  lower_bound FLOAT64,
  upper_bound FLOAT64,
  confidence FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY model_type;

-- ZL Predictions 6m
CREATE TABLE IF NOT EXISTS `cbi-v15.forecasts.zl_predictions_6m` (
  date DATE,
  model_type STRING,
  prediction FLOAT64,
  lower_bound FLOAT64,
  upper_bound FLOAT64,
  confidence FLOAT64
)
PARTITION BY DATE(date)
CLUSTER BY model_type;

