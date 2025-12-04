-- MotherDuck Schema: ops
-- Operational metadata, QC, tests, logs, pipeline metrics

-- Data quality reports (AnoFox Tabular outputs)
CREATE TABLE IF NOT EXISTS ops.data_quality_zl_daily (
    report_id UUID PRIMARY KEY,
    report_date DATE NOT NULL,
    table_name VARCHAR NOT NULL,
    
    -- Quality metrics
    total_rows BIGINT,
    null_count BIGINT,
    null_ratio DECIMAL(6, 4),
    duplicate_count INT,
    
    -- Gap analysis
    gap_count INT,
    max_gap_days INT,
    gaps_filled INT,
    
    -- Outlier analysis
    outlier_count INT,
    outliers_removed INT,
    outlier_method VARCHAR,  -- 'zscore', 'isolation_forest'
    
    -- Distribution stats
    mean_val DECIMAL(12, 4),
    std_val DECIMAL(12, 4),
    min_val DECIMAL(12, 4),
    max_val DECIMAL(12, 4),
    skewness DECIMAL(6, 4),
    kurtosis DECIMAL(6, 4),
    
    -- Status
    status VARCHAR,  -- 'PASSED', 'WARNING', 'FAILED'
    issues JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline test runs
CREATE TABLE IF NOT EXISTS ops.pipeline_test_runs (
    test_id UUID PRIMARY KEY,
    test_type VARCHAR NOT NULL,  -- 'extension_load', 'synthetic_forecast', 'data_quality'
    run_timestamp TIMESTAMP NOT NULL,
    
    -- Test details
    test_name VARCHAR,
    test_description TEXT,
    
    -- Results
    status VARCHAR NOT NULL,  -- 'passed', 'failed', 'warning'
    execution_time_seconds DECIMAL(10, 2),
    error_message TEXT,
    
    -- Metadata
    tested_component VARCHAR,  -- 'anofox_forecast', 'neural_lstm', 'xgboost_meta'
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ingestion completion log
CREATE TABLE IF NOT EXISTS ops.ingestion_completion (
    ingestion_id UUID PRIMARY KEY,
    source VARCHAR NOT NULL,  -- 'databento', 'fred', 'scrapecreators'
    bucket VARCHAR,  -- 'fx', 'rates', 'trump'
    run_id VARCHAR NOT NULL,
    
    -- Period
    start_date DATE,
    end_date DATE,
    row_count BIGINT,
    
    -- Status
    status VARCHAR NOT NULL,  -- 'success', 'partial', 'failed'
    error_message TEXT,
    
    -- Timing
    started_at TIMESTAMP,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline metrics daily (composite health score)
CREATE TABLE IF NOT EXISTS ops.pipeline_metrics_daily (
    metric_date DATE PRIMARY KEY,
    
    -- Test metrics
    test_pass_rate DECIMAL(5, 4),
    tests_run INT,
    tests_passed INT,
    tests_failed INT,
    
    -- Data quality
    data_quality_score DECIMAL(5, 4),
    tables_checked INT,
    tables_passed INT,
    
    -- Model performance
    model_edge_vs_baseline DECIMAL(6, 2),  -- Average improvement
    ensemble_mape DECIMAL(10, 6),
    
    -- Regime
    regime_classification VARCHAR,
    regime_confidence DECIMAL(5, 4),
    
    -- Composite score
    composite_pipeline_score DECIMAL(5, 4),
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Anomaly flags (from AnoFox Tabular)
CREATE TABLE IF NOT EXISTS ops.anomaly_flags_zl (
    flag_id UUID PRIMARY KEY,
    date DATE NOT NULL,
    table_name VARCHAR NOT NULL,
    column_name VARCHAR,
    
    -- Anomaly details
    anomaly_type VARCHAR,  -- 'outlier', 'gap', 'regime_shift', 'schema_drift'
    anomaly_score DECIMAL(10, 6),
    anomaly_method VARCHAR,  -- 'isolation_forest', 'zscore'
    
    -- Action taken
    action VARCHAR,  -- 'flagged', 'removed', 'adjusted'
    
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

