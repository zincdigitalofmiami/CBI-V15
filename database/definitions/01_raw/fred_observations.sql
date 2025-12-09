-- FRED Observations
-- Stores time series data for FRED economic series

CREATE TABLE IF NOT EXISTS raw.fred_observations (
    -- Composite Primary Key
    series_id TEXT,
    date DATE,
    
    -- Value
    value DOUBLE,
    
    -- Metadata
    realtime_start DATE,
    realtime_end DATE,
    
    -- Ingestion Tracking
    created_at TIMESTAMP DEFAULT current_timestamp,
    
    PRIMARY KEY (series_id, date)
);

-- Index for time series queries
CREATE INDEX IF NOT EXISTS idx_fred_obs_series_date ON raw.fred_observations(series_id, date);

-- Index for date range queries
CREATE INDEX IF NOT EXISTS idx_fred_obs_date ON raw.fred_observations(date);

