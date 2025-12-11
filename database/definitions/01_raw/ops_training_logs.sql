-- Operations: Training Logs
-- Stores immediate audit results after each bucket training
-- Populated by src/reporting/training_auditor.py (hot-audit loop)

CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS ops.training_logs (
    bucket_name VARCHAR NOT NULL,  -- crush, china, fx, fed, tariff, biofuel, energy, volatility, main
    trained_at TIMESTAMP NOT NULL,
    best_model VARCHAR,  -- AutoGluon model name (e.g., WeightedEnsemble_L2)
    ag_score DOUBLE,  -- AutoGluon test score (pinball_loss for quantile)
    baseline_score DOUBLE,  -- AnoFox Structure baseline (naive persistence)
    lift DOUBLE,  -- (baseline - ag_score) / baseline (% improvement)
    fit_time_seconds DOUBLE,  -- Training duration
    model_count INTEGER,  -- Number of models trained
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (bucket_name, trained_at)
);

-- Index for queries
CREATE INDEX IF NOT EXISTS idx_training_logs_trained_at 
ON ops.training_logs (trained_at DESC);

-- Example query: Latest training results
-- SELECT * FROM ops.training_logs 
-- WHERE trained_at >= CURRENT_DATE - INTERVAL '7 days'
-- ORDER BY trained_at DESC;



