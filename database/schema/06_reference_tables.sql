-- Reference Tables: Feature Catalog & Model Registry

-- 1. Feature Catalog
CREATE TABLE IF NOT EXISTS reference.feature_catalog (
    feature_name      TEXT PRIMARY KEY,
    bucket            TEXT NOT NULL,
    topic_group       TEXT,
    data_type         TEXT NOT NULL,
    source_table      TEXT NOT NULL,
    source_expression TEXT,
    description       TEXT,
    is_active         BOOLEAN NOT NULL DEFAULT TRUE,
    is_experimental   BOOLEAN NOT NULL DEFAULT FALSE,
    created_by        TEXT NOT NULL DEFAULT 'anofox',
    created_at        TIMESTAMP NOT NULL DEFAULT current_timestamp,
    updated_by        TEXT,
    updated_at        TIMESTAMP,
    tags              TEXT
);

-- Index for bucket exploration
CREATE INDEX IF NOT EXISTS idx_feature_catalog_bucket ON reference.feature_catalog (bucket);

-- 2. Model Registry
CREATE TABLE IF NOT EXISTS reference.model_registry (
    model_id             TEXT PRIMARY KEY,
    version              TEXT NOT NULL,
    horizon              TEXT NOT NULL,
    symbol               TEXT NOT NULL,
    engine_type          TEXT NOT NULL,
    feature_set_name     TEXT NOT NULL,
    training_table       TEXT NOT NULL,
    target_column        TEXT NOT NULL,
    regime_scope         TEXT NOT NULL DEFAULT 'all',
    status               TEXT NOT NULL DEFAULT 'candidate',
    owner                TEXT NOT NULL DEFAULT 'anofox',
    metric_primary_name  TEXT,
    metric_primary_value DOUBLE,
    metric_secondary     TEXT,
    created_at           TIMESTAMP NOT NULL DEFAULT current_timestamp,
    updated_at           TIMESTAMP,
    notes                TEXT
);

-- Indexes for model lookup
CREATE INDEX IF NOT EXISTS idx_model_registry_symbol_horizon ON reference.model_registry (symbol, horizon);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON reference.model_registry (status);
