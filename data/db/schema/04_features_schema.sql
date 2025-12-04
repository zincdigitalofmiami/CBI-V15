-- MotherDuck Schema: features
-- Engineered features (276 → pruned to 80-120)

-- ZL features (engineered technical indicators + derivations)
CREATE TABLE IF NOT EXISTS features.zl_features (
    date DATE PRIMARY KEY,
    
    -- Price features (from databento)
    databento_zl_close DECIMAL(10, 2),
    databento_zl_volume BIGINT,
    databento_zl_open_interest BIGINT,
    
    -- Technical indicators (calculated locally or via AnoFox statistics)
    tech_sma_5 DECIMAL(10, 2),
    tech_sma_20 DECIMAL(10, 2),
    tech_sma_50 DECIMAL(10, 2),
    tech_sma_200 DECIMAL(10, 2),
    tech_rsi_14 DECIMAL(5, 2),
    tech_volatility_21d DECIMAL(10, 6),
    tech_trend_strength_60d DECIMAL(5, 4),
    
    -- Macro features (from FRED)
    fred_dxy DECIMAL(10, 4),
    fred_vix DECIMAL(10, 4),
    fred_treasury_10y DECIMAL(6, 4),
    fred_fed_funds DECIMAL(6, 4),
    
    -- Correlations (rolling windows)
    corr_zl_wti_90d DECIMAL(6, 4),
    corr_zl_brl_90d DECIMAL(6, 4),
    corr_zl_palm_90d DECIMAL(6, 4),
    
    -- Weather features
    weather_us_iowa_prcp_mm DECIMAL(6, 2),
    weather_argentina_drought_zscore DECIMAL(6, 4),
    weather_br_mato_grosso_tavg_c DECIMAL(5, 2),
    
    -- Policy features
    policy_trump_score DECIMAL(5, 4),
    eia_rin_price_d4 DECIMAL(10, 4),
    
    -- Positioning
    cftc_managed_money_netlong BIGINT,
    
    -- (Total: 80-120 production features after pruning)
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature importance log (AnoFox analysis results)
CREATE TABLE IF NOT EXISTS features.feature_importance_log (
    run_id UUID NOT NULL,
    feature_name VARCHAR NOT NULL,
    correlation_with_zl_returns DECIMAL(6, 4),
    anofox_importance_score DECIMAL(10, 6),
    importance_rank INT,
    is_selected BOOLEAN DEFAULT FALSE,
    pruning_rationale TEXT,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, feature_name)
);

