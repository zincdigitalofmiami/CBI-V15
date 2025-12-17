-- Raw Databento Futures OHLCV - HOURLY (1H)
-- Segmented table for intraday analysis
-- Uses TIMESTAMP instead of DATE for granularity

CREATE TABLE IF NOT EXISTS raw.databento_futures_ohlcv_1h (
    symbol VARCHAR NOT NULL,
    ts_event TIMESTAMP NOT NULL,  -- Hourly timestamp (e.g. 2025-01-01 14:00:00)
    open DECIMAL(10, 4),          -- Higher precision for intraday
    high DECIMAL(10, 4),
    low DECIMAL(10, 4),
    close DECIMAL(10, 4),
    volume BIGINT,
    open_interest BIGINT,         -- Often NULL intraday, but kept for schema parity
    source VARCHAR DEFAULT 'databento_1h',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, ts_event)
);

-- Indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_databento_1h_symbol 
    ON raw.databento_futures_ohlcv_1h(symbol);
    
CREATE INDEX IF NOT EXISTS idx_databento_1h_ts 
    ON raw.databento_futures_ohlcv_1h(ts_event);


