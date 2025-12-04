-- Reference Tables: Calendars, Regimes, Splits, Feature Catalog, Model Registry

-- Trading Calendar
CREATE TABLE IF NOT EXISTS reference.trading_calendar (
    as_of_date        DATE PRIMARY KEY,
    is_trading_day    BOOLEAN NOT NULL DEFAULT TRUE,
    exchange          TEXT DEFAULT 'CME',
    notes             TEXT
);

-- Regime Calendar
CREATE TABLE IF NOT EXISTS reference.regime_calendar (
    as_of_date        DATE PRIMARY KEY,
    regime            TEXT NOT NULL,  -- 'bull', 'bear', 'neutral', 'volatile'
    regime_score      DOUBLE,
    created_at        TIMESTAMP DEFAULT current_timestamp
);

-- Train/Val/Test Splits
CREATE TABLE IF NOT EXISTS reference.train_val_test_splits (
    as_of_date        DATE PRIMARY KEY,
    split_label       TEXT NOT NULL,  -- 'train', 'val', 'test'
    split_version     TEXT DEFAULT 'v1'
);

-- Feature Catalog
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

CREATE INDEX IF NOT EXISTS idx_feature_catalog_bucket ON reference.feature_catalog (bucket);

-- Model Registry
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

CREATE INDEX IF NOT EXISTS idx_model_registry_symbol_horizon ON reference.model_registry (symbol, horizon);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON reference.model_registry (status);
