-- MotherDuck Schema: training
-- ML-ready matrices, backtest results, model training logs

-- Canonical daily ML matrix (276 features → pruned to 80-120)
CREATE TABLE IF NOT EXISTS training.daily_ml_matrix (
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    
    -- All production features (with prefixes)
    -- Databento
    databento_zl_close DECIMAL(10, 2),
    databento_zl_volume BIGINT,
    databento_zl_open_interest BIGINT,
    
    -- FRED (55+ series)
    fred_dxy DECIMAL(10, 4),
    fred_vix DECIMAL(10, 4),
    fred_treasury_10y DECIMAL(6, 4),
    -- ... all 55+ FRED series
    
    -- Weather (granular by region)
    weather_us_iowa_tavg_c DECIMAL(5, 2),
    weather_us_iowa_prcp_mm DECIMAL(6, 2),
    weather_argentina_drought_zscore DECIMAL(6, 4),
    -- ... all weather regions
    
    -- EIA
    eia_rin_price_d4 DECIMAL(10, 4),
    eia_biodiesel_prod_padd2 DOUBLE,
    -- ... all EIA series
    
    -- USDA
    usda_wasde_world_soyoil_prod DOUBLE,
    usda_exports_soybeans_net_sales_china DOUBLE,
    -- ... all USDA series
    
    -- CFTC
    cftc_managed_money_netlong BIGINT,
    -- ... all CFTC metrics
    
    -- Policy
    policy_trump_score DECIMAL(5, 4),
    policy_trump_expected_zl_move DECIMAL(6, 4),
    -- ... all policy features
    
    -- Technical
    tech_sma_5 DECIMAL(10, 2),
    tech_sma_20 DECIMAL(10, 2),
    tech_rsi_14 DECIMAL(5, 2),
    tech_volatility_21d DECIMAL(10, 6),
    -- ... all technical indicators
    
    -- Correlations
    corr_zl_wti_90d DECIMAL(6, 4),
    corr_zl_brl_90d DECIMAL(6, 4),
    -- ... all correlations
    
    -- Targets (forward returns for each horizon)
    target_1w DECIMAL(10, 6),
    target_1m DECIMAL(10, 6),
    target_3m DECIMAL(10, 6),
    target_6m DECIMAL(10, 6),
    target_12m DECIMAL(10, 6),
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

-- Model backtest results
CREATE TABLE IF NOT EXISTS training.model_backtest_zl (
    backtest_id UUID PRIMARY KEY,
    backtest_date DATE NOT NULL,
    model_tier VARCHAR NOT NULL,  -- 'anofox', 'neural', 'meta'
    model_name VARCHAR NOT NULL,
    horizon VARCHAR NOT NULL,  -- '1W', '1M', '3M', '6M', '12M'
    
    -- Validation period
    validation_start_date DATE,
    validation_end_date DATE,
    n_predictions INT,
    
    -- Performance metrics
    mae DECIMAL(10, 6),
    rmse DECIMAL(10, 6),
    mape DECIMAL(10, 6),
    bias DECIMAL(10, 6),
    coverage_90 DECIMAL(5, 4),
    coverage_95 DECIMAL(5, 4),
    directional_accuracy DECIMAL(5, 4),
    sharpe_ratio DECIMAL(6, 4),
    
    -- Beat baseline
    vs_naive_improvement DECIMAL(6, 2),
    vs_autoets_improvement DECIMAL(6, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- All 45 model training runs
CREATE TABLE IF NOT EXISTS training.zl_model_runs (
    run_id UUID PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL,
    model_tier VARCHAR NOT NULL,  -- 'anofox', 'neural', 'meta'
    model_name VARCHAR NOT NULL,
    driver_group_id VARCHAR,  -- Links to reference.driver_group
    horizon INT NOT NULL,  -- Days (5, 21, 63, 126, 252)
    
    -- Hyperparameters (JSON for flexibility)
    hyperparameters JSON,
    
    -- Training metadata
    training_time_seconds DECIMAL(10, 2),
    training_rows INT,
    validation_rows INT,
    
    -- Performance metrics
    mae DECIMAL(10, 6),
    rmse DECIMAL(10, 6),
    mape DECIMAL(10, 6),
    smape DECIMAL(10, 6),
    mase DECIMAL(10, 6),
    bias DECIMAL(10, 6),
    coverage_90 DECIMAL(5, 4),
    coverage_95 DECIMAL(5, 4),
    
    -- Ensemble selection
    is_selected BOOLEAN DEFAULT FALSE,
    ensemble_weight DECIMAL(5, 4),
    
    -- Status
    status VARCHAR DEFAULT 'completed',  -- 'running', 'completed', 'failed'
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

