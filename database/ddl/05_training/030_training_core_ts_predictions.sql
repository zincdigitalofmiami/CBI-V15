-- Training: Core TS Predictions
-- L1 TimeSeriesPredictor outputs

CREATE TABLE IF NOT EXISTS training.core_ts_predictions (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,
    prediction_type VARCHAR NOT NULL,  -- 'oof' or 'live'
    -- Quantile predictions
    q10 DECIMAL(10, 4),
    q25 DECIMAL(10, 4),
    q50 DECIMAL(10, 4),  -- Median (point forecast)
    q75 DECIMAL(10, 4),
    q90 DECIMAL(10, 4),
    -- Derived
    iqr DECIMAL(10, 4),  -- q75 - q25
    prediction_interval_width DECIMAL(10, 4),  -- q90 - q10
    -- Model metadata
    model_name VARCHAR DEFAULT 'TimeSeriesPredictor',
    model_version VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code, prediction_type)
);

-- Core TS uses ALL features + optionally specialist signals as covariates

