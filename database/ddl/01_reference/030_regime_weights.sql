-- Regime-specific Model Weights
-- Ensemble weights vary by market regime

CREATE TABLE IF NOT EXISTS reference.regime_weights (
    regime VARCHAR NOT NULL,  -- 'CALM', 'STRESSED', 'CRISIS'
    horizon_code VARCHAR NOT NULL,  -- '1w', '1m', '3m', '6m'
    bucket_name VARCHAR NOT NULL,  -- 'crush', 'china', 'fx', etc.
    weight DECIMAL(5, 4) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (regime, horizon_code, bucket_name)
);

-- Crisis regimes typically weight Volatility bucket higher
-- Calm regimes weight Crush/Biofuel fundamentals higher

