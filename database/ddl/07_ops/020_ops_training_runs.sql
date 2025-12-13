-- Ops: Training Runs
-- Model training run metadata

CREATE TABLE IF NOT EXISTS ops.training_runs (
    run_id VARCHAR PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL,
    model_tier VARCHAR NOT NULL,  -- 'specialist', 'core', 'meta', 'ensemble'
    model_name VARCHAR NOT NULL,
    bucket VARCHAR,  -- For specialists
    horizon_code VARCHAR,
    -- Training config
    training_start_date DATE,
    training_end_date DATE,
    n_train_rows INT,
    n_val_rows INT,
    hyperparameters JSON,
    -- Performance
    val_mape DECIMAL(10, 6),
    val_rmse DECIMAL(10, 6),
    val_directional_accuracy DECIMAL(5, 4),
    -- Timing
    training_time_seconds DECIMAL(10, 2),
    -- Status
    status VARCHAR NOT NULL,  -- 'running', 'completed', 'failed'
    error_message TEXT,
    -- Artifact
    artifact_uri VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_runs_model 
    ON ops.training_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_training_runs_status 
    ON ops.training_runs(status);

