-- Ops: Data Quality Log
-- Daily data quality reports

CREATE TABLE IF NOT EXISTS ops.data_quality_log (
    report_id VARCHAR PRIMARY KEY,
    report_date DATE NOT NULL,
    schema_name VARCHAR NOT NULL,
    table_name VARCHAR NOT NULL,
    -- Quality metrics
    total_rows BIGINT,
    null_count BIGINT,
    null_ratio DECIMAL(6, 4),
    duplicate_count INT,
    -- Freshness
    max_date DATE,
    days_stale INT,
    -- Gap analysis
    gap_count INT,
    max_gap_days INT,
    -- Distribution
    mean_val DECIMAL(12, 4),
    std_val DECIMAL(12, 4),
    min_val DECIMAL(12, 4),
    max_val DECIMAL(12, 4),
    -- Status
    status VARCHAR,  -- 'PASSED', 'WARNING', 'FAILED'
    issues JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dq_log_table 
    ON ops.data_quality_log(table_name);
CREATE INDEX IF NOT EXISTS idx_dq_log_status 
    ON ops.data_quality_log(status);

