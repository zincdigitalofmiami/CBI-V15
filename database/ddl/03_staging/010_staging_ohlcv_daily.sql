-- Staging: OHLCV Daily
-- Cleaned multi-symbol market data

CREATE TABLE IF NOT EXISTS staging.ohlcv_daily (
    symbol VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    open_interest BIGINT,
    -- Quality flags
    gap_filled BOOLEAN DEFAULT FALSE,
    outlier_adjusted BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, as_of_date)
);

-- Covers all 33 canonical symbols
-- Cleaned: gaps forward-filled, outliers winsorized

CREATE INDEX IF NOT EXISTS idx_staging_ohlcv_symbol 
    ON staging.ohlcv_daily(symbol);
CREATE INDEX IF NOT EXISTS idx_staging_ohlcv_as_of_date 
    ON staging.ohlcv_daily(as_of_date);

