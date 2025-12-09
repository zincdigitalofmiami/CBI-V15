-- Training Tables: Features + Targets + Splits

-- Main Training Matrix
CREATE TABLE IF NOT EXISTS training.daily_ml_matrix_zl (
    -- Keys
    as_of_date          DATE NOT NULL,
    symbol              TEXT NOT NULL,
    
    -- Regime & Splits (LOCKED)
    regime              TEXT,
    training_weight     DOUBLE,
    train_val_test_split TEXT,  -- 'train', 'val', 'test'
    
    -- Targets (LOCKED)
    target_ret_1w       DOUBLE,
    target_ret_1m       DOUBLE,
    target_ret_3m       DOUBLE,
    target_ret_6m       DOUBLE,
    target_ret_12m      DOUBLE,
    
    -- Big 8 Bucket Scores (LOCKED)
    crush_bucket_score    DOUBLE,
    china_bucket_score    DOUBLE,
    fx_bucket_score       DOUBLE,
    fed_bucket_score      DOUBLE,
    tariff_bucket_score   DOUBLE,
    biofuel_bucket_score  DOUBLE,
    energy_bucket_score   DOUBLE,
    volatility_bucket_score DOUBLE,
    
    -- Big 8 Neural Scores (LOCKED)
    crush_neural_score    DOUBLE,
    china_neural_score    DOUBLE,
    fx_neural_score       DOUBLE,
    fed_neural_score      DOUBLE,
    tariff_neural_score   DOUBLE,
    biofuel_neural_score  DOUBLE,
    energy_neural_score   DOUBLE,
    volatility_neural_score DOUBLE,
    
    -- Master Neural Score (LOCKED)
    master_neural_score   DOUBLE,
    
    -- Raw features (FREE - AnoFox can add columns)
    
    PRIMARY KEY (as_of_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_training_date ON training.daily_ml_matrix_zl (as_of_date);
CREATE INDEX IF NOT EXISTS idx_training_split ON training.daily_ml_matrix_zl (train_val_test_split);
