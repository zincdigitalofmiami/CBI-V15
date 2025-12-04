-- Feature Tables: Denormalized feature matrices

-- Main V15 Feature Matrix (Big 8 + all raw features)
CREATE TABLE IF NOT EXISTS features.daily_ml_matrix_zl_v15 (
    -- Keys
    as_of_date    DATE NOT NULL,
    symbol        TEXT NOT NULL,
    
    -- Regime
    regime        TEXT,
    
    -- Big 8 Bucket Scores
    crush_bucket_score    DOUBLE,
    china_bucket_score    DOUBLE,
    fx_bucket_score       DOUBLE,
    fed_bucket_score      DOUBLE,
    tariff_bucket_score   DOUBLE,
    biofuel_bucket_score  DOUBLE,
    energy_bucket_score   DOUBLE,
    vol_bucket_score      DOUBLE,
    
    -- Big 8 Neural Scores
    crush_neural_score    DOUBLE,
    china_neural_score    DOUBLE,
    fx_neural_score       DOUBLE,
    fed_neural_score      DOUBLE,
    tariff_neural_score   DOUBLE,
    biofuel_neural_score  DOUBLE,
    energy_neural_score   DOUBLE,
    vol_neural_score      DOUBLE,
    
    -- Master Neural Score
    master_neural_score   DOUBLE,
    
    -- Raw features added dynamically by AnoFox
    -- (AnoFox can ALTER TABLE to add columns)
    
    PRIMARY KEY (as_of_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_features_date ON features.daily_ml_matrix_zl_v15 (as_of_date);
