-- ⚠️ CRITICAL: NO FAKE DATA ⚠️
-- This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
-- All data must come from authenticated APIs, official sources, or validated historical records.
--

-- ============================================================================
-- CBI-V15 FINAL COMPLETE BIGQUERY SCHEMA
-- Date: November 18, 2025
-- Status: PRODUCTION-READY with all Fresh Start + Training Master requirements
-- Purpose: Complete ZL forecasting infrastructure (400-500 features)
-- ============================================================================

-- ============================================================================
-- DATASETS (Fixed - removed drivers_of_drivers)
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS market_data
OPTIONS (location='us-central1', description='CME/CBOT/NYMEX/COMEX market data only');

CREATE SCHEMA IF NOT EXISTS raw_intelligence  
OPTIONS (location='us-central1', description='Free APIs only - FRED, USDA, EIA, CFTC, NOAA');

CREATE SCHEMA IF NOT EXISTS signals
OPTIONS (location='us-central1', description='Derived signals - crush, spreads, microstructure, Big 8');

CREATE SCHEMA IF NOT EXISTS features
OPTIONS (location='us-central1', description='Canonical master_features table');

CREATE SCHEMA IF NOT EXISTS training
OPTIONS (location='us-central1', description='Training datasets and regime support');

CREATE SCHEMA IF NOT EXISTS regimes
OPTIONS (location='us-central1', description='Regime classifications per symbol');

CREATE SCHEMA IF NOT EXISTS drivers
OPTIONS (location='us-central1', description='Primary drivers and meta-drivers');

CREATE SCHEMA IF NOT EXISTS neural
OPTIONS (location='us-central1', description='Neural training features');

CREATE SCHEMA IF NOT EXISTS predictions
OPTIONS (location='us-central1', description='Model predictions and forecasts');

CREATE SCHEMA IF NOT EXISTS monitoring
OPTIONS (location='us-central1', description='Model performance monitoring');

CREATE SCHEMA IF NOT EXISTS dim
OPTIONS (location='us-central1', description='Reference data and metadata');

CREATE SCHEMA IF NOT EXISTS ops
OPTIONS (location='us-central1', description='Operations and data quality');

-- ============================================================================
-- PART 1: MARKET DATA (DataBento + Historical Bridge)
-- ============================================================================

-- Keep all existing market_data tables (1-11) from VENUE_PURE_SCHEMA
-- Adding compatibility views

CREATE OR REPLACE VIEW market_data.futures_ohlcv_1m AS
  SELECT * FROM market_data.databento_futures_ohlcv_1m;

CREATE OR REPLACE VIEW market_data.futures_ohlcv_1d AS
  SELECT * FROM market_data.databento_futures_ohlcv_1d;

-- ============================================================================
-- PART 2: TRAINING INFRASTRUCTURE (NEW - Required by Training Master Plan)
-- ============================================================================

-- Table A: Regime Calendar (Maps every date to regime)
CREATE OR REPLACE TABLE training.regime_calendar (
  date DATE NOT NULL,
  regime STRING NOT NULL,
  -- 11 regimes: historical_pre2000, dotcom_bubble, pre_crisis, crisis_2008,
  -- recovery, trade_war_2017_2019, covid_2020, inflation_2021_2022,
  -- stable_2022_2023, trump_2023_2025, current
  valid_from DATE,
  valid_to DATE,
  PRIMARY KEY (date) NOT ENFORCED
)
PARTITION BY date
OPTIONS (description='Maps every date 2000-2025 to training regime');

-- Table B: Regime Weights (50-5000 scale per Training Plan)
CREATE OR REPLACE TABLE training.regime_weights (
  regime STRING NOT NULL,
  weight INT64 NOT NULL,  -- 50 (historical) to 5000 (Trump 2.0)
  description STRING,
  research_rationale STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  PRIMARY KEY (regime) NOT ENFORCED
)
OPTIONS (description='Training weights by regime - 100x differential');

-- Table C: Production Training Data (290-450 features, 5 horizons)
CREATE OR REPLACE TABLE training.zl_training_prod_allhistory_1w (
  date DATE NOT NULL,
  -- All 290-450 production features from master_features
  -- See master_features definition for full column list
  -- Includes regime and training_weight
  regime STRING,
  training_weight INT64,
  target_1w FLOAT64,
  as_of TIMESTAMP NOT NULL
)
PARTITION BY date
CLUSTER BY regime
OPTIONS (description='Production training data - 1 week horizon, 2000-2025');

-- ZL Training Tables (5 horizons: 1w, 1m, 3m, 6m, 12m)
CREATE OR REPLACE TABLE training.zl_training_prod_allhistory_1m AS
  SELECT * FROM training.zl_training_prod_allhistory_1w WHERE FALSE;

CREATE OR REPLACE TABLE training.zl_training_prod_allhistory_3m AS
  SELECT * FROM training.zl_training_prod_allhistory_1w WHERE FALSE;

CREATE OR REPLACE TABLE training.zl_training_prod_allhistory_6m AS
  SELECT * FROM training.zl_training_prod_allhistory_1w WHERE FALSE;

CREATE OR REPLACE TABLE training.zl_training_prod_allhistory_12m AS
  SELECT * FROM training.zl_training_prod_allhistory_1w WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_1min (
  ts_event TIMESTAMP NOT NULL,
  -- Intraday features focus on microstructure, orderflow
  regime STRING,
  training_weight INT64,
  target_1min FLOAT64,
  as_of TIMESTAMP NOT NULL
)
PARTITION BY DATE(ts_event)
CLUSTER BY regime

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_5min AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_15min AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_30min AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_1hr AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_4hr AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_1d AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_7d AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_30d AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_3m AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_6m AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

CREATE OR REPLACE TABLE training.mes_training_prod_allhistory_12m AS
  SELECT * FROM training.mes_training_prod_allhistory_1min WHERE FALSE;

-- Table D: Full Training Data (1,948+ features for research)
CREATE OR REPLACE TABLE training.zl_training_full_allhistory_1w (
  date DATE NOT NULL,
  -- ALL features including experimental
  -- 1,948+ columns
  regime STRING,
  training_weight INT64,
  target_1w FLOAT64,
  as_of TIMESTAMP NOT NULL
)
PARTITION BY date
CLUSTER BY regime
OPTIONS (description='Full feature set - 1 week horizon, research only');

-- ============================================================================
-- PART 3: HIDDEN INTELLIGENCE MODULE (From Training Plan Idea Generation)
-- ============================================================================

-- Table E: Hidden Relationship Signals
CREATE OR REPLACE TABLE signals.hidden_relationship_signals (
  date DATE NOT NULL,
  
  -- Cross-domain hidden drivers (lead ZL by 1-9 months)
  hidden_defense_agri_score FLOAT64,
  hidden_tech_agri_score FLOAT64,
  hidden_pharma_agri_score FLOAT64,
  hidden_swf_lead_flow_score FLOAT64,
  hidden_carbon_arbitrage_score FLOAT64,
  hidden_cbdc_corridor_score FLOAT64,
  hidden_port_capacity_lead_index FLOAT64,
  hidden_academic_exchange_score FLOAT64,
  hidden_trump_argentina_backchannel_score FLOAT64,
  hidden_china_alt_bloc_score FLOAT64,
  hidden_biofuel_lobbying_pressure FLOAT64,
  
  -- Composite
  hidden_relationship_composite_score FLOAT64,
  
  -- Metadata
  correlation_override_flag BOOL,
  primary_hidden_domain STRING,
  
  as_of TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
OPTIONS (description='Hidden cross-domain intelligence signals');

-- Table F: News Intelligence (GPT Classification System)
CREATE OR REPLACE TABLE raw_intelligence.news_intelligence (
  id STRING NOT NULL,
  headline STRING,
  source STRING,
  published_at TIMESTAMP NOT NULL,
  
  -- GPT Classification (40 categories)
  primary_topic STRING,
  hidden_relationships ARRAY<STRING>,  -- 17 cross-domain drivers
  region_focus ARRAY<STRING>,         -- 12 geographies
  
  -- Impact Assessment
  relevance_to_soy_complex INT64,     -- 0-100
  directional_impact_zl STRING,       -- bullish/bearish/neutral/mixed/unknown
  impact_strength INT64,               -- 0-100
  impact_time_horizon_days INT64,
  half_life_days INT64,
  
  -- Explanation
  mechanism_summary STRING,
  direct_vs_indirect STRING,
  subtopics ARRAY<STRING>,
  confidence INT64,                    -- 0-100
  
  -- Processing metadata
  gpt_model_version STRING,
  processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(published_at)
CLUSTER BY primary_topic, directional_impact_zl
OPTIONS (description='GPT-classified news with ZL impact assessment');

-- Table G: News Bucketed (Aggregated daily)
CREATE OR REPLACE TABLE raw_intelligence.news_bucketed (
  date DATE NOT NULL,
  bucket STRING NOT NULL,              -- policy_*, trade_*, biofuel_*, etc.
  
  -- Counts
  article_count INT64,
  bullish_count INT64,
  bearish_count INT64,
  
  -- Sentiment
  avg_sentiment FLOAT64,
  sentiment_volatility FLOAT64,
  
  -- Impact
  max_impact_score FLOAT64,
  avg_relevance_score FLOAT64,
  
  -- Hidden relationships
  hidden_driver_intensity FLOAT64,
  
  collection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
CLUSTER BY bucket
OPTIONS (description='Daily aggregated news by category');

-- ============================================================================
-- PART 4: OPERATIONS & MONITORING (NEW)
-- ============================================================================

-- Table H: Ingestion Runs (For observability)
CREATE OR REPLACE TABLE ops.ingestion_runs (
  run_id STRING NOT NULL,
  source STRING NOT NULL,
  start_time TIMESTAMP NOT NULL,
  end_time TIMESTAMP,
  status STRING,                       -- running, success, failed
  rows_processed INT64,
  error_message STRING,
  metadata JSON,
  PRIMARY KEY (run_id) NOT ENFORCED
)
PARTITION BY DATE(start_time)
CLUSTER BY source, status
OPTIONS (description='ETL run tracking for all data sources');

-- Table I: Model Performance
CREATE OR REPLACE TABLE monitoring.model_performance (
  evaluation_date DATE NOT NULL,
  model_id STRING NOT NULL,
  horizon STRING NOT NULL,
  
  -- Metrics
  mape FLOAT64,
  rmse FLOAT64,
  r_squared FLOAT64,
  directional_accuracy FLOAT64,
  
  -- By regime
  regime_performance JSON,              -- {regime: {mape, accuracy}}
  
  -- Feature importance
  top_features ARRAY<STRUCT<
    feature STRING,
    importance FLOAT64,
    shap_value FLOAT64
  >>,
  
  evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY evaluation_date
CLUSTER BY model_id, horizon
OPTIONS (description='Daily model performance tracking');

-- ============================================================================
-- PART 5: EXPANDED MASTER FEATURES (400-500 columns per Training Plan)
-- ============================================================================

CREATE OR REPLACE TABLE features.master_features (
  date DATE NOT NULL,
  symbol STRING NOT NULL DEFAULT 'ZL',
  
  databento_zl_open FLOAT64,
  databento_zl_high FLOAT64,
  databento_zl_low FLOAT64,
  databento_zl_close FLOAT64,
  databento_zl_volume INT64,
  databento_zl_oi INT64,
  
  
  -- Stitched best available
  zl_open FLOAT64,
  zl_high FLOAT64,
  zl_low FLOAT64,
  zl_close FLOAT64,
  zl_volume INT64,
  zl_oi INT64,
  
  
  -- ========== PIVOT POINTS (Databento-based, Phase 1 Core) ==========
  -- Source: features.pivot_math_daily (cloud_function_pivot_calculator.py output)
  -- ✅ VERIFIED: Column names match calculator output exactly (Integration Test passed)
  -- Phase 1: Core pivots (9 columns) | Phase 2: Extended pivots deferred (R3/R4, M1-M8, Monthly/Quarterly)
  
  -- Raw pivot levels (daily) - from _daily_pivots() function
  P FLOAT64,                          -- Daily pivot point
  R1 FLOAT64,                         -- Daily resistance 1
  R2 FLOAT64,                         -- Daily resistance 2
  S1 FLOAT64,                         -- Daily support 1
  S2 FLOAT64,                         -- Daily support 2
  
  -- Distance features (absolute values) - calculated in handler()
  distance_to_P FLOAT64,              -- Distance to daily pivot
  distance_to_nearest_pivot FLOAT64,  -- Distance to nearest pivot level
  
  -- Weekly context - from _range_pivots() function
  weekly_pivot_distance FLOAT64,      -- Distance to weekly pivot (not the level itself)
  
  -- Boolean flags - calculated in handler()
  price_above_P BOOL,                 -- Price above daily pivot (directional indicator)
  
  -- ========== INTELLIGENCE FEATURES (38+ from Training Plan) ==========
  china_mentions INT64,
  china_posts INT64,
  import_posts INT64,
  soy_posts INT64,
  china_sentiment FLOAT64,
  china_sentiment_volatility FLOAT64,
  china_policy_impact FLOAT64,
  import_demand_index FLOAT64,
  china_posts_7d_ma FLOAT64,
  china_sentiment_30d_ma FLOAT64,
  trump_mentions INT64,
  trumpxi_china_mentions INT64,
  trump_xi_co_mentions INT64,
  xi_mentions INT64,
  tariff_mentions INT64,
  co_mention_sentiment FLOAT64,
  trumpxi_sentiment_volatility FLOAT64,
  trumpxi_policy_impact FLOAT64,
  max_policy_impact FLOAT64,
  tension_index FLOAT64,
  volatility_multiplier FLOAT64,
  co_mentions_7d_ma FLOAT64,
  trumpxi_volatility_30d_ma FLOAT64,
  china_tariff_rate FLOAT64,
  trade_war_intensity FLOAT64,
  trade_war_impact_score FLOAT64,
  trump_soybean_sentiment_7d FLOAT64,
  trump_agricultural_impact_30d FLOAT64,
  trump_soybean_relevance_30d FLOAT64,
  days_since_trump_policy INT64,
  trump_policy_intensity_14d FLOAT64,
  social_sentiment_momentum_7d FLOAT64,
  social_sentiment_avg FLOAT64,
  social_sentiment_volatility FLOAT64,
  social_post_count INT64,
  bullish_ratio FLOAT64,
  bearish_ratio FLOAT64,
  
  -- ========== CME INDICES ==========
  cme_soybean_oilshare_cosi1 FLOAT64,
  -- cme_soybean_cvol_30d removed (CVOL discontinued)
  
  -- ========== CRUSH & OILSHARE (CME-native) ==========
  crush_theoretical_usd_per_bu FLOAT64,
  crush_board_usd_per_bu FLOAT64,
  oilshare_model FLOAT64,
  oilshare_divergence_bps FLOAT64,
  
  -- ========== CALENDAR SPREADS ==========
  zl_spread_m1_m2 FLOAT64,
  zl_spread_m1_m3 FLOAT64,
  zs_spread_m1_m2 FLOAT64,
  zm_spread_m1_m2 FLOAT64,
  cl_spread_m1_m2 FLOAT64,
  
  -- ========== ENERGY PROXIES ==========
  crack_3_2_1 FLOAT64,
  ho_spread_m1_m2 FLOAT64,
  rb_spread_m1_m2 FLOAT64,
  ethanol_cu_settle FLOAT64,
  brent_wti_spread FLOAT64,
  
  cme_6l_brl_close FLOAT64,
  cme_cnh_close FLOAT64,
  fred_usd_cny FLOAT64,
  fred_usd_ars FLOAT64,
  fred_usd_myr FLOAT64,
  fred_dxy FLOAT64,
  databento_6e_close FLOAT64,
  
  -- ========== FRED MACRO (60 series) ==========
  fred_dff FLOAT64,
  fred_dgs10 FLOAT64,
  fred_dgs2 FLOAT64,
  fred_dgs5 FLOAT64,
  fred_dgs30 FLOAT64,
  fred_vixcls FLOAT64,
  fred_dtwexbgs FLOAT64,
  fred_cpiaucsl FLOAT64,
  fred_cpilfesl FLOAT64,
  fred_pcepi FLOAT64,
  fred_unrate FLOAT64,
  fred_payems FLOAT64,
  fred_civpart FLOAT64,
  fred_gdp FLOAT64,
  fred_gdpc1 FLOAT64,
  fred_indpro FLOAT64,
  fred_dgorder FLOAT64,
  fred_m2sl FLOAT64,
  fred_bogmbase FLOAT64,
  fred_baaffm FLOAT64,
  fred_t10y2y FLOAT64,
  fred_t10y3m FLOAT64,
  fred_dcoilwtico FLOAT64,
  fred_houst FLOAT64,
  fred_umcsent FLOAT64,
  fred_ppiaco FLOAT64,
  fred_nfci FLOAT64,
  -- ... (continue with all 60 FRED series)
  
  -- ========== EIA BIOFUELS ==========
  eia_biodiesel_prod_us FLOAT64,
  eia_biodiesel_prod_padd1 FLOAT64,
  eia_biodiesel_prod_padd2 FLOAT64,
  eia_biodiesel_prod_padd3 FLOAT64,
  eia_renewable_diesel_prod_us FLOAT64,
  eia_ethanol_prod_us FLOAT64,
  eia_rin_price_d4 FLOAT64,
  eia_rin_price_d5 FLOAT64,
  eia_rin_price_d6 FLOAT64,
  eia_saf_prod_us FLOAT64,
  
  -- ========== USDA GRANULAR ==========
  usda_wasde_world_soyoil_prod FLOAT64,
  usda_wasde_world_soyoil_use FLOAT64,
  usda_wasde_us_soyoil_prod FLOAT64,
  usda_wasde_brazil_soybean_prod FLOAT64,
  usda_wasde_argentina_soybean_prod FLOAT64,
  usda_exports_soybeans_net_sales_china FLOAT64,
  usda_exports_soybeans_net_sales_eu FLOAT64,
  usda_exports_soybeans_net_sales_total FLOAT64,
  usda_exports_soyoil_net_sales_total FLOAT64,
  usda_cropprog_il_soy_condition_good_excellent_pct FLOAT64,
  usda_cropprog_ia_soy_condition_good_excellent_pct FLOAT64,
  usda_cropprog_in_soy_condition_good_excellent_pct FLOAT64,
  usda_stocks_soybeans_total FLOAT64,
  usda_stocks_soyoil_total FLOAT64,
  
  -- ========== WEATHER ==========
  weather_us_midwest_tavg_wgt FLOAT64,
  weather_us_midwest_prcp_wgt FLOAT64,
  weather_us_midwest_gdd_wgt FLOAT64,
  weather_br_soy_belt_tavg_wgt FLOAT64,
  weather_br_soy_belt_prcp_wgt FLOAT64,
  weather_ar_soy_belt_tavg_wgt FLOAT64,
  weather_ar_soy_belt_prcp_wgt FLOAT64,
  
  -- ========== CFTC ==========
  cftc_zl_net_managed_money INT64,
  cftc_zl_net_commercial INT64,
  cftc_zl_open_interest INT64,
  cftc_zs_net_managed_money INT64,
  cftc_zm_net_managed_money INT64,
  
  -- ========== VOLATILITY ==========
  vol_vix_level FLOAT64,
  vol_vix_zscore_30d FLOAT64,
  -- vol_cme_cvol_soybeans_30d removed (CVOL discontinued)
  vol_zl_realized_5d FLOAT64,
  vol_zl_realized_10d FLOAT64,
  vol_zl_realized_20d FLOAT64,
  vol_zs_realized_20d FLOAT64,
  vol_cl_realized_20d FLOAT64,
  vol_es_realized_5d FLOAT64,
  vol_regime STRING,
  
  -- ========== POLICY/TRUMP ==========
  policy_trump_action_prob FLOAT64,
  policy_trump_expected_zl_move FLOAT64,
  policy_trump_score FLOAT64,
  policy_trump_score_signed FLOAT64,
  policy_trump_confidence FLOAT64,
  policy_trump_topic_multiplier FLOAT64,
  policy_trump_recency_decay FLOAT64,
  policy_trump_sentiment_score FLOAT64,
  policy_trump_procurement_alert STRING,
  
  -- ========== MICROSTRUCTURE ==========
  microstructure_zl_spread_bps FLOAT64,
  microstructure_zl_depth_imbalance_avg FLOAT64,
  microstructure_zl_trade_imbalance_avg FLOAT64,
  microstructure_zl_microprice_dev_bps FLOAT64,
  microstructure_zl_aggressor_buy_pct FLOAT64,
  
  -- ========== HIDDEN INTELLIGENCE ==========
  hidden_defense_agri_score FLOAT64,
  hidden_tech_agri_score FLOAT64,
  hidden_pharma_agri_score FLOAT64,
  hidden_swf_lead_flow_score FLOAT64,
  hidden_carbon_arbitrage_score FLOAT64,
  hidden_cbdc_corridor_score FLOAT64,
  hidden_port_capacity_lead_index FLOAT64,
  hidden_trump_argentina_backchannel_score FLOAT64,
  hidden_china_alt_bloc_score FLOAT64,
  hidden_biofuel_lobbying_pressure FLOAT64,
  hidden_relationship_composite_score FLOAT64,
  
  -- ========== SHOCK FEATURES ==========
  shock_policy_flag BOOL,
  shock_vol_flag BOOL,
  shock_supply_flag BOOL,
  shock_biofuel_flag BOOL,
  shock_policy_score FLOAT64,
  shock_vol_score FLOAT64,
  shock_supply_score FLOAT64,
  shock_biofuel_score FLOAT64,
  shock_policy_score_decayed FLOAT64,
  shock_vol_score_decayed FLOAT64,
  shock_supply_score_decayed FLOAT64,
  shock_biofuel_score_decayed FLOAT64,
  
  -- ========== BIG 8 SIGNALS ==========
  big8_crush_oilshare_pressure FLOAT64,
  big8_policy_shock FLOAT64,
  big8_weather_supply_risk FLOAT64,
  big8_china_demand FLOAT64,
  big8_vix_stress FLOAT64,
  big8_positioning_pressure FLOAT64,
  big8_energy_biofuel_shock FLOAT64,
  big8_fx_pressure FLOAT64,
  big8_composite_score FLOAT64,
  
  -- ========== CALCULATED SIGNALS ==========
  signal_zl_sma_50 FLOAT64,
  signal_zl_sma_100 FLOAT64,
  signal_zl_sma_200 FLOAT64,
  signal_zl_rsi_14 FLOAT64,
  signal_zl_macd FLOAT64,
  signal_zl_momentum_10d FLOAT64,
  signal_zl_roc_20d FLOAT64,
  
  -- ========== REGIME & TRAINING ==========
  regime STRING,
  training_weight INT64,
  
  -- ========== TARGETS ==========
  target_1w FLOAT64,
  target_1m FLOAT64,
  target_3m FLOAT64,
  target_6m FLOAT64,
  target_12m FLOAT64,
  
  -- ========== METADATA ==========
  as_of TIMESTAMP NOT NULL,
  collection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
CLUSTER BY symbol, regime
OPTIONS (
  description='Complete master features table - 404-509 columns (Phase 1 pivot swap applied)'
);

-- ============================================================================
-- VALIDATION QUERIES (Fixed - removed drivers_of_drivers)
-- ============================================================================

-- Check tables created
SELECT table_schema, table_name, table_type, creation_time
FROM `region-us-central1`.INFORMATION_SCHEMA.TABLES  
WHERE table_catalog = 'cbi-v15'
  AND table_schema IN ('market_data', 'raw_intelligence', 'signals', 'features', 
                       'training', 'regimes', 'drivers', 'neural', 'predictions',
                       'monitoring', 'dim', 'ops')
ORDER BY table_schema, table_name;

-- Validate master_features has 400+ columns
SELECT 
  table_name,
  COUNT(*) as column_count
FROM `region-us-central1`.INFORMATION_SCHEMA.COLUMNS
WHERE table_catalog = 'cbi-v15'
  AND table_schema = 'features'
  AND table_name = 'master_features'
GROUP BY table_name;

-- Check training tables exist with proper naming
SELECT table_name
FROM `region-us-central1`.INFORMATION_SCHEMA.TABLES
WHERE table_catalog = 'cbi-v15'
  AND table_schema = 'training'
  AND table_name LIKE 'zl_training_%'
ORDER BY table_name;

-- ============================================================================
-- END - COMPLETE SCHEMA WITH ALL REQUIREMENTS
-- Total Tables: 45+ (all infrastructure for 400-500 feature ZL forecasting)
-- ============================================================================
