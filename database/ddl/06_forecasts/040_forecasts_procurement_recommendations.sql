-- Forecasts: Procurement Recommendations
-- BUY/WAIT/LADDER decisions for US Oil Solutions

CREATE TABLE IF NOT EXISTS forecasts.procurement_recommendations (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,
    -- Recommendation
    action VARCHAR NOT NULL,  -- 'BUY', 'WAIT', 'LADDER'
    action_confidence DECIMAL(5, 4),
    -- Price levels
    current_price DECIMAL(10, 4),
    expected_price DECIMAL(10, 4),
    target_price DECIMAL(10, 4),
    stop_price DECIMAL(10, 4),
    -- Expected value analysis
    expected_savings_pct DECIMAL(6, 4),
    expected_savings_usd DECIMAL(12, 2),  -- Based on typical procurement size
    -- Risk analysis
    downside_risk_pct DECIMAL(6, 4),
    upside_potential_pct DECIMAL(6, 4),
    risk_reward_ratio DECIMAL(6, 4),
    -- Tripwires (trigger levels for re-evaluation)
    tripwire_up DECIMAL(10, 4),
    tripwire_down DECIMAL(10, 4),
    -- Regime context
    regime VARCHAR,
    volatility_level VARCHAR,  -- 'LOW', 'NORMAL', 'HIGH'
    -- Narrative
    recommendation_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code)
);

-- Decision engine converts forecasts → actionable procurement guidance

