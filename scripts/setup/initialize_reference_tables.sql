-- Initialize Reference Tables
-- Run after creating skeleton tables

-- ============================================================================
-- REGIME CALENDAR
-- ============================================================================

CREATE OR REPLACE TABLE `cbi-v15.reference.regime_calendar` (
  regime_type STRING,
  start_date DATE,
  end_date DATE,
  description STRING,
  base_weight FLOAT64,
  vix_multiplier FLOAT64
)
CLUSTER BY regime_type;

INSERT INTO `cbi-v15.reference.regime_calendar` VALUES
('trump_2018', DATE('2018-01-01'), DATE('2020-12-31'), 'Trump first term trade war era', 1.0, 1.5),
('trump_2024', DATE('2024-01-01'), DATE('2025-12-31'), 'Trump second term', 1.0, 1.3),
('normal', DATE('2010-01-01'), DATE('2017-12-31'), 'Normal market conditions', 1.0, 1.0),
('crisis', DATE('2020-03-01'), DATE('2020-06-30'), 'COVID crisis', 1.0, 2.0);

-- ============================================================================
-- REGIME WEIGHTS (VIX-Based)
-- ============================================================================

CREATE OR REPLACE TABLE `cbi-v15.reference.regime_weights` (
  regime_type STRING,
  base_weight FLOAT64,
  vix_multiplier FLOAT64,
  shock_multiplier_policy FLOAT64,
  shock_multiplier_vol FLOAT64,
  shock_multiplier_supply FLOAT64,
  shock_multiplier_geopol FLOAT64
)
CLUSTER BY regime_type;

INSERT INTO `cbi-v15.reference.regime_weights` VALUES
('trump_2018', 1.0, 1.5, 0.15, 0.15, 0.15, 0.15),
('trump_2024', 1.0, 1.3, 0.15, 0.15, 0.15, 0.15),
('normal', 1.0, 1.0, 0.15, 0.15, 0.15, 0.15),
('crisis', 1.0, 2.0, 0.15, 0.15, 0.15, 0.15);

-- ============================================================================
-- TRAIN/VAL/TEST SPLITS
-- ============================================================================

CREATE OR REPLACE TABLE `cbi-v15.reference.train_val_test_splits` (
  set_name STRING,
  start_date DATE,
  end_date DATE,
  description STRING
);

INSERT INTO `cbi-v15.reference.train_val_test_splits` VALUES
('train', DATE('2010-01-01'), DATE('2018-12-31'), 'Training set (9 years)'),
('val', DATE('2019-01-01'), DATE('2021-12-31'), 'Validation set (3 years)'),
('test', DATE('2022-01-01'), DATE('2025-12-31'), 'Test set (4 years)');

-- ============================================================================
-- NEURAL DRIVERS (Layer 3 → Layer 2 → Layer 1)
-- ============================================================================

CREATE OR REPLACE TABLE `cbi-v15.reference.neural_drivers` (
  layer INT64,
  driver_name STRING,
  description STRING,
  input_features ARRAY<STRING>,
  output_feature STRING
)
CLUSTER BY layer, driver_name;

INSERT INTO `cbi-v15.reference.neural_drivers` VALUES
(3, 'dollar_deep', 'Deep drivers for dollar (rate spreads, risk sentiment, capital flows)', ['fred_dtwexbgs', 'fred_dgs10', 'fred_dgs2'], 'dollar_neural_score'),
(3, 'fed_deep', 'Deep drivers for Fed (employment, inflation, financial conditions)', ['fred_unrate', 'fred_cpi', 'fred_fedfunds'], 'fed_neural_score'),
(3, 'crush_deep', 'Deep drivers for crush (processing economics, demand, logistics)', ['board_crush', 'oil_share', 'hog_spread'], 'crush_neural_score'),
(2, 'dollar_neural', 'Neural composite signal for dollar', ['dollar_neural_score'], 'dollar_neural_score'),
(2, 'fed_neural', 'Neural composite signal for Fed', ['fed_neural_score'], 'fed_neural_score'),
(2, 'crush_neural', 'Neural composite signal for crush', ['crush_neural_score'], 'crush_neural_score'),
(1, 'master_neural', 'Master neural score combining all signals', ['dollar_neural_score', 'fed_neural_score', 'crush_neural_score'], 'neural_master_score');

-- ============================================================================
-- INGESTION COMPLETION TRACKING
-- ============================================================================

CREATE OR REPLACE TABLE `cbi-v15.ops.ingestion_completion` (
  date DATE,
  source STRING,
  completed_at TIMESTAMP,
  status STRING,
  rows_ingested INT64,
  error_message STRING
)
PARTITION BY DATE(date)
CLUSTER BY source;

-- Initialize with today's date (no completions yet)
INSERT INTO `cbi-v15.ops.ingestion_completion` VALUES
(CURRENT_DATE(), 'databento_zl', NULL, 'pending', NULL, NULL),
(CURRENT_DATE(), 'databento_other', NULL, 'pending', NULL, NULL),
(CURRENT_DATE(), 'fred', NULL, 'pending', NULL, NULL),
(CURRENT_DATE(), 'scrapecreators_news', NULL, 'pending', NULL, NULL),
(CURRENT_DATE(), 'scrapecreators_trump', NULL, 'pending', NULL, NULL),
(CURRENT_DATE(), 'usda', NULL, 'pending', NULL, NULL),
(CURRENT_DATE(), 'cftc', NULL, 'pending', NULL, NULL),
(CURRENT_DATE(), 'eia', NULL, 'pending', NULL, NULL),
(CURRENT_DATE(), 'weather', NULL, 'pending', NULL, NULL);

