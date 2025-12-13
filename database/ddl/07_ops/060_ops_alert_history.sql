-- Ops: Alert History
-- System alerts and notifications

CREATE TABLE IF NOT EXISTS ops.alert_history (
    alert_id VARCHAR PRIMARY KEY,
    alert_type VARCHAR NOT NULL,  -- 'data_quality', 'model_degradation', 'ingestion_failure', 'forecast_anomaly'
    severity VARCHAR NOT NULL,  -- 'info', 'warning', 'error', 'critical'
    -- Context
    source VARCHAR,
    model_name VARCHAR,
    horizon_code VARCHAR,
    -- Message
    title VARCHAR NOT NULL,
    message TEXT,
    -- Status
    status VARCHAR DEFAULT 'open',  -- 'open', 'acknowledged', 'resolved'
    resolved_at TIMESTAMP,
    resolved_by VARCHAR,
    -- Timing
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_status 
    ON ops.alert_history(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity 
    ON ops.alert_history(severity);

