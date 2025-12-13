-- Feature Catalog
-- Registry of all features with metadata

CREATE TABLE IF NOT EXISTS reference.feature_catalog (
    feature_name VARCHAR PRIMARY KEY,
    feature_group VARCHAR NOT NULL,  -- 'technical', 'fundamental', 'sentiment', 'macro'
    bucket VARCHAR,  -- Big 8 bucket: 'crush', 'china', 'fx', 'fed', 'tariff', 'biofuel', 'energy', 'volatility'
    data_source VARCHAR NOT NULL,  -- 'databento', 'fred', 'cftc', 'usda', etc.
    description TEXT,
    unit VARCHAR,
    frequency VARCHAR DEFAULT 'daily',  -- 'daily', 'weekly', 'monthly'
    lag_days INT DEFAULT 0,  -- Publication lag
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for bucket filtering
CREATE INDEX IF NOT EXISTS idx_feature_catalog_bucket 
    ON reference.feature_catalog(bucket);

