-- Staging: Daily Returns
-- Log returns and derived metrics

CREATE TABLE IF NOT EXISTS staging.daily_returns (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    close DECIMAL(10, 2),
    log_return DECIMAL(10, 6),
    simple_return DECIMAL(10, 6),
    -- Rolling volatility
    volatility_5d DECIMAL(10, 6),
    volatility_21d DECIMAL(10, 6),
    volatility_63d DECIMAL(10, 6),
    -- Volume metrics
    volume_zscore DECIMAL(6, 4),
    oi_change BIGINT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);

-- Log returns: ln(close_t / close_t-1)
-- Volatility: rolling std of log returns

CREATE INDEX IF NOT EXISTS idx_staging_returns_symbol 
    ON staging.daily_returns(symbol);

