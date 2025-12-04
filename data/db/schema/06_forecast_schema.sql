-- MotherDuck Schema: forecast
-- Model outputs, ensemble results, production forecasts

-- AnoFox 31-model baseline panel (The "Big Run")
CREATE TABLE IF NOT EXISTS forecast.zl_anofox_panel (
    as_of_date DATE NOT NULL,
    engine_type VARCHAR NOT NULL,  -- 'price', 'volatility', 'macro'
    model_name VARCHAR NOT NULL,
    horizon_days INT NOT NULL,  -- 5, 21, 63, 126, 252
    horizon_code VARCHAR NOT NULL,  -- '1W', '1M', '3M', '6M', '12M'
    
    -- Forecast values
    point_forecast DECIMAL(10, 4) NOT NULL,
    lower_90 DECIMAL(10, 4),
    upper_90 DECIMAL(10, 4),
    lower_95 DECIMAL(10, 4),
    upper_95 DECIMAL(10, 4),
    
    -- Metrics (computed during backtest)
    mae DECIMAL(10, 6),
    mape DECIMAL(10, 6),
    
    -- Residuals (for meta-learning)
    residuals JSON,
    
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, engine_type, model_name, horizon_code)
);

-- Neural model panel (7 models)
CREATE TABLE IF NOT EXISTS forecast.zl_neural_panel (
    as_of_date DATE NOT NULL,
    model_name VARCHAR NOT NULL,  -- 'LSTM', 'GRU', 'TCN', 'TFT', 'GARCH-LSTM', 'HAR-LSTM', 'N-BEATS'
    horizon_days INT NOT NULL,
    horizon_code VARCHAR NOT NULL,
    
    -- Forecast values
    point_forecast DECIMAL(10, 4) NOT NULL,
    lower_90 DECIMAL(10, 4),
    upper_90 DECIMAL(10, 4),
    lower_95 DECIMAL(10, 4),
    upper_95 DECIMAL(10, 4),
    
    -- Neural-specific outputs
    attention_weights JSON,  -- TFT attention
    decomposition JSON,  -- N-BEATS trend/seasonal blocks
    
    -- Performance
    mae DECIMAL(10, 6),
    mape DECIMAL(10, 6),
    beats_baseline_by_pct DECIMAL(6, 2),  -- Must be ≥5% to be used
    
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, model_name, horizon_code)
);

-- Final ensemble output (production forecast)
CREATE TABLE IF NOT EXISTS forecast.zl_ensemble_output (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,  -- '1W', '1M', '3M', '6M', '12M'
    target_date DATE NOT NULL,
    
    -- Ensemble forecast
    point_forecast DECIMAL(10, 4) NOT NULL,
    lower_90 DECIMAL(10, 4),
    upper_90 DECIMAL(10, 4),
    lower_95 DECIMAL(10, 4),
    upper_95 DECIMAL(10, 4),
    
    -- Ensemble metadata
    active_models JSON,  -- List of 5-7 models with weights
    regime VARCHAR NOT NULL,  -- 'CALM', 'STRESSED', 'CRISIS'
    model_confidence DECIMAL(5, 4),
    
    -- Driver contributions (SHAP-aggregated)
    biofuel_energy_contribution DECIMAL(8, 4),
    trade_tariff_contribution DECIMAL(8, 4),
    weather_supply_contribution DECIMAL(8, 4),
    palm_substitution_contribution DECIMAL(8, 4),
    macro_risk_contribution DECIMAL(8, 4),
    positioning_contribution DECIMAL(8, 4),
    policy_regulation_contribution DECIMAL(8, 4),
    technical_regime_contribution DECIMAL(8, 4),
    
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code)
);

-- Ensemble config (active model weights)
CREATE TABLE IF NOT EXISTS forecast.ensemble_config (
    horizon_code VARCHAR NOT NULL,
    regime VARCHAR NOT NULL,  -- 'CALM', 'STRESSED', 'CRISIS'
    model_tier VARCHAR NOT NULL,  -- 'anofox', 'neural', 'meta'
    model_name VARCHAR NOT NULL,
    ensemble_weight DECIMAL(5, 4) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (horizon_code, regime, model_name)
);

