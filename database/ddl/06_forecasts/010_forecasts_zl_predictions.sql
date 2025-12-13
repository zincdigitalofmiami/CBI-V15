-- Forecasts: ZL Predictions
-- Final ensemble output (serving contract)

CREATE TABLE IF NOT EXISTS forecasts.zl_predictions (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,  -- '1w', '1m', '3m', '6m'
    target_date DATE NOT NULL,
    -- Final quantiles
    q10 DECIMAL(10, 4),
    q25 DECIMAL(10, 4),
    q50 DECIMAL(10, 4),  -- Point forecast
    q75 DECIMAL(10, 4),
    q90 DECIMAL(10, 4),
    -- Direction
    direction INT,  -- -1, 0, 1
    direction_probability DECIMAL(5, 4),
    -- Confidence
    prediction_confidence DECIMAL(5, 4),
    model_agreement DECIMAL(5, 4),
    -- Regime
    regime VARCHAR,
    -- Metadata
    model_version VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code)
);

-- This is the SERVING CONTRACT for the dashboard
-- Dashboard reads ONLY from forecasts.* schema

CREATE INDEX IF NOT EXISTS idx_forecasts_zl_date 
    ON forecasts.zl_predictions(as_of_date DESC);

