-- MotherDuck Schema: signals
-- Live scoring, driver groups, production forecasts for dashboard

-- Big 8 live signals (15-minute updates)
CREATE TABLE IF NOT EXISTS signals.big_eight_live (
    timestamp TIMESTAMP PRIMARY KEY,
    
    -- 8 driver group scores
    biofuel_energy_score DECIMAL(5, 4),
    trade_tariff_score DECIMAL(5, 4),
    weather_supply_score DECIMAL(5, 4),
    palm_substitution_score DECIMAL(5, 4),
    macro_risk_score DECIMAL(5, 4),
    positioning_score DECIMAL(5, 4),
    policy_regulation_score DECIMAL(5, 4),
    technical_regime_score DECIMAL(5, 4),
    
    -- Composite procurement index
    procurement_sentiment_index DECIMAL(6, 4),
    
    -- Regime
    current_regime VARCHAR,  -- 'CALM', 'STRESSED', 'CRISIS'
    
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Driver group scores (daily aggregation)
CREATE TABLE IF NOT EXISTS signals.driver_group_score_daily (
    date DATE NOT NULL,
    driver_group_id VARCHAR NOT NULL,
    score DECIMAL(5, 4),
    sentiment VARCHAR,  -- 'bullish', 'neutral', 'bearish'
    zscore DECIMAL(6, 4),
    percentile_rank DECIMAL(5, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, driver_group_id)
);

-- Driver group contributions (SHAP-aggregated by group)
CREATE TABLE IF NOT EXISTS signals.driver_group_contribution (
    date DATE NOT NULL,
    horizon VARCHAR NOT NULL,
    driver_group_id VARCHAR NOT NULL,
    
    -- Attribution sources
    shap_contribution DECIMAL(8, 4),
    ols_beta DECIMAL(8, 4),
    tft_attention DECIMAL(5, 4),
    
    -- Aggregated
    total_contribution DECIMAL(8, 4),
    contribution_rank INT,
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, horizon, driver_group_id)
);

