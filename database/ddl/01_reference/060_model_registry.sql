-- Model Registry
-- Tracks all trained models with versioning and performance

CREATE TABLE IF NOT EXISTS reference.model_registry (
    model_id VARCHAR PRIMARY KEY,
    model_tier VARCHAR NOT NULL,  -- 'specialist' (L0), 'core' (L1), 'meta' (L2), 'ensemble' (L3)
    model_name VARCHAR NOT NULL,
    bucket VARCHAR,  -- For specialists: 'crush', 'china', etc.
    horizon_code VARCHAR,  -- '1w', '1m', '3m', '6m'
    
    -- Performance (latest validation)
    mape DECIMAL(10, 6),
    directional_accuracy DECIMAL(5, 4),
    coverage_90 DECIMAL(5, 4),
    
    -- Ensemble config
    ensemble_weight DECIMAL(5, 4),
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Artifact location
    artifact_uri VARCHAR,  -- S3/R2/Volume path
    
    -- Versioning
    version INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

