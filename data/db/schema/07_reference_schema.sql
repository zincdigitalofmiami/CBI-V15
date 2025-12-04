-- MotherDuck Schema: reference
-- Slowly changing reference tables

-- Driver groups (8 groups for Chris's dashboard)
CREATE TABLE IF NOT EXISTS reference.driver_group (
    driver_group_id VARCHAR PRIMARY KEY,
    driver_group_name VARCHAR NOT NULL,
    description TEXT,
    dashboard_page VARCHAR,  -- Which page displays this group
    sentiment_layer INT,  -- Maps to 9-layer sentiment system
    update_frequency VARCHAR  -- '15min', 'daily', 'weekly'
);

-- Seed driver groups
INSERT INTO reference.driver_group VALUES
('BIOFUEL_ENERGY', 'Biofuel & Energy', 'RINs, EPA mandates, biodiesel demand, crude correlation', 'Dashboard,Strategy', 2, '15min'),
('TRADE_TARIFF', 'Trade & Tariffs', 'Trump, China, Brazil, Argentina trade relations', 'Trade Intelligence', 3, '15min'),
('WEATHER_SUPPLY', 'Weather & Supply', 'Drought, La Niña, harvest updates, USDA reports', 'Sentiment', 4, 'daily'),
('PALM_SUBSTITUTION', 'Palm Substitution', 'Indonesia levy, Malaysia stocks, palm spread', 'Strategy', 5, 'daily'),
('MACRO_RISK', 'Macro Risk', 'VIX, DXY, Fed Funds, Treasury yields', 'Dashboard', 7, '15min'),
('POSITIONING', 'Positioning', 'CFTC COT, managed money flows', 'Sentiment', 9, 'weekly'),
('POLICY_REGULATION', 'Policy & Regulation', 'EPA, USDA, lobbying, executive orders', 'Trade Intelligence', 2, 'daily'),
('TECHNICAL_REGIME', 'Technical & Regime', 'Volatility regime, technical indicators', 'Dashboard', 8, '15min')
ON CONFLICT DO NOTHING;

-- Feature to driver group mapping
CREATE TABLE IF NOT EXISTS reference.feature_to_driver_group_map (
    feature_name VARCHAR PRIMARY KEY,
    driver_group_id VARCHAR NOT NULL REFERENCES reference.driver_group(driver_group_id),
    feature_description TEXT,
    data_source VARCHAR,  -- 'fred', 'databento', 'weather', etc.
    mapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model registry (active production models)
CREATE TABLE IF NOT EXISTS reference.model_registry (
    model_id VARCHAR PRIMARY KEY,
    model_tier VARCHAR NOT NULL,  -- 'anofox', 'neural', 'meta'
    model_name VARCHAR NOT NULL,
    driver_group_id VARCHAR REFERENCES reference.driver_group(driver_group_id),
    horizon INT,  -- Days (5, 21, 63, 126, 252)
    
    -- Performance (latest validation)
    mape DECIMAL(10, 6),
    directional_accuracy DECIMAL(5, 4),
    
    -- Ensemble config
    ensemble_weight DECIMAL(5, 4),
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Artifact location
    artifact_path VARCHAR,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Regime calendar (historical regime assignments)
CREATE TABLE IF NOT EXISTS reference.regime_calendar (
    date DATE PRIMARY KEY,
    regime VARCHAR NOT NULL,  -- 'CALM', 'STRESSED', 'CRISIS'
    vol_zscore DECIMAL(6, 4),
    garch_variance DECIMAL(10, 6),
    vix_level DECIMAL(6, 2),
    regime_confidence DECIMAL(5, 4),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

