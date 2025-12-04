-- Ops Tables: Ingestion Status and Operational Metrics

-- Ingestion Status
CREATE TABLE IF NOT EXISTS ops.ingestion_status (
    ingestion_id      BIGINT PRIMARY KEY,
    source_name       TEXT NOT NULL,  -- 'scrape_creator', 'databento'
    bucket_name       TEXT,
    started_at        TIMESTAMP NOT NULL DEFAULT current_timestamp,
    finished_at       TIMESTAMP,
    status            TEXT NOT NULL,  -- 'running', 'success', 'failed'
    rows_ingested     BIGINT,
    error_message     TEXT,
    created_by        TEXT DEFAULT 'anofox'
);

-- Pipeline Metrics
CREATE TABLE IF NOT EXISTS ops.pipeline_metrics (
    metric_id         BIGINT PRIMARY KEY,
    pipeline_name     TEXT NOT NULL,  -- 'build_features', 'build_training', 'build_forecasts'
    metric_name       TEXT NOT NULL,
    metric_value      DOUBLE,
    as_of_date        DATE,
    created_at        TIMESTAMP DEFAULT current_timestamp
);

CREATE INDEX IF NOT EXISTS idx_ops_ingestion_source ON ops.ingestion_status (source_name);
CREATE INDEX IF NOT EXISTS idx_ops_metrics_pipeline ON ops.pipeline_metrics (pipeline_name);
