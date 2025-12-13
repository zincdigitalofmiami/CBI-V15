-- Ops: Model Performance
-- Rolling performance metrics

CREATE TABLE IF NOT EXISTS ops.model_performance (
    as_of_date DATE NOT NULL,
    model_name VARCHAR NOT NULL,
    horizon_code VARCHAR NOT NULL,
    -- Rolling metrics (30-day)
    rolling_mape_30d DECIMAL(10, 6),
    rolling_rmse_30d DECIMAL(10, 6),
    rolling_directional_accuracy_30d DECIMAL(5, 4),
    rolling_sharpe_30d DECIMAL(6, 4),
    -- Vs baseline
    vs_naive_improvement DECIMAL(6, 2),
    vs_ets_improvement DECIMAL(6, 2),
    -- Stability
    prediction_stability DECIMAL(5, 4),  -- How much forecasts change day-to-day
    -- Status
    is_degraded BOOLEAN DEFAULT FALSE,  -- Performance below threshold
    degradation_reason TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, model_name, horizon_code)
);

-- Monitor for model drift and degradation

