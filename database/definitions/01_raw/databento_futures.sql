-- Databento Futures OHLCV Data
-- Daily bars for all 38 futures symbols

CREATE TABLE IF NOT EXISTS raw.databento_futures (
    -- Composite Primary Key
    symbol TEXT,
    timestamp TIMESTAMP,
    
    -- OHLCV
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT current_timestamp,
    
    PRIMARY KEY (symbol, timestamp)
);

-- Index for symbol queries
CREATE INDEX IF NOT EXISTS idx_databento_symbol ON raw.databento_futures(symbol, timestamp);

-- Index for date range queries
CREATE INDEX IF NOT EXISTS idx_databento_timestamp ON raw.databento_futures(timestamp);

