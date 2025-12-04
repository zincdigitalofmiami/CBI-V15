-- Forecast Tables: Model predictions

-- V15 Forecasts by horizon
CREATE TABLE IF NOT EXISTS forecasts.zl_v15_predictions (
    as_of_date          DATE NOT NULL,
    symbol              TEXT NOT NULL,
    model_id            TEXT NOT NULL,
    run_id              BIGINT,
    horizon             TEXT NOT NULL,  -- '1w', '1m', '3m', '6m', '12m'
    
    -- Predictions
    y_pred              DOUBLE,
    y_pred_lower        DOUBLE,
    y_pred_upper        DOUBLE,
    
    -- Context at prediction time
    master_neural_score_at_run DOUBLE,
    regime_at_run       TEXT,
    
    -- Metadata
    created_at          TIMESTAMP DEFAULT current_timestamp,
    
    PRIMARY KEY (as_of_date, symbol, model_id, horizon)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts.zl_v15_predictions (as_of_date);
CREATE INDEX IF NOT EXISTS idx_forecasts_model ON forecasts.zl_v15_predictions (model_id);
