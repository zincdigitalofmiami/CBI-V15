-- Ops: Ingestion Completion
-- Tracks successful data ingestion runs

CREATE TABLE IF NOT EXISTS ops.ingestion_completion (
    ingestion_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,  -- 'databento', 'fred', 'cftc', 'usda', etc.
    job_name VARCHAR,
    run_id VARCHAR NOT NULL,
    -- Period covered
    start_date DATE,
    end_date DATE,
    row_count BIGINT,
    -- Status
    status VARCHAR NOT NULL,  -- 'success', 'partial', 'failed'
    error_message TEXT,
    -- Timing
    started_at TIMESTAMP,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_seconds DECIMAL(10, 2)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_source 
    ON ops.ingestion_completion(source);
CREATE INDEX IF NOT EXISTS idx_ingestion_status 
    ON ops.ingestion_completion(status);

