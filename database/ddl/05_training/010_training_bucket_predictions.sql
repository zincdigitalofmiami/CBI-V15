-- Training: Bucket Predictions
-- L0 specialist outputs (OOF and live)

CREATE TABLE IF NOT EXISTS training.bucket_predictions (
    as_of_date DATE NOT NULL,
    bucket VARCHAR NOT NULL,  -- 'crush', 'china', etc.
    horizon_code VARCHAR NOT NULL,  -- '1w', '1m', '3m', '6m'
    prediction_type VARCHAR NOT NULL,  -- 'oof' (out-of-fold) or 'live'
    -- Probabilistic outputs
    p_up DECIMAL(5, 4),
    p_down DECIMAL(5, 4),
    expected_return DECIMAL(10, 6),
    confidence DECIMAL(5, 4),
    -- Quantiles
    q10 DECIMAL(10, 4),
    q50 DECIMAL(10, 4),
    q90 DECIMAL(10, 4),
    -- Model metadata
    model_version VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, bucket, horizon_code, prediction_type)
);

-- OOF predictions used for meta-model training
-- Live predictions used for inference

CREATE INDEX IF NOT EXISTS idx_bucket_predictions_bucket 
    ON training.bucket_predictions(bucket);
