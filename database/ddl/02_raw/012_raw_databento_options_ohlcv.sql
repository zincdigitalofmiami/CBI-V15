-- Raw Databento Options OHLCV - Daily
-- Options contracts for all futures with listed options (ZL, ZS, ZC, CL, GC, etc.)
-- Schema: GLBX.MDP3 ohlcv-1d with options symbology

CREATE TABLE IF NOT EXISTS raw.databento_options_ohlcv_1d (
    symbol VARCHAR NOT NULL,              -- Underlying root (ZL, ZS, CL, etc.)
    contract_symbol VARCHAR NOT NULL,     -- Full options symbol from Databento
    as_of_date DATE NOT NULL,             -- Trading date
    strike_price DECIMAL(12, 4),          -- Strike price
    expiration_date DATE,                 -- Contract expiration
    option_type VARCHAR(4),               -- 'C' (call) or 'P' (put)
    open DECIMAL(10, 4),
    high DECIMAL(10, 4),
    low DECIMAL(10, 4),
    close DECIMAL(10, 4),
    volume BIGINT,
    open_interest BIGINT,
    implied_volatility DECIMAL(8, 4),    -- If available from Databento
    delta DECIMAL(8, 4),                  -- Greeks if available
    gamma DECIMAL(8, 4),
    theta DECIMAL(8, 4),
    vega DECIMAL(8, 4),
    source VARCHAR DEFAULT 'databento_options',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (contract_symbol, as_of_date)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_options_symbol 
    ON raw.databento_options_ohlcv_1d(symbol);
CREATE INDEX IF NOT EXISTS idx_options_date 
    ON raw.databento_options_ohlcv_1d(as_of_date);
CREATE INDEX IF NOT EXISTS idx_options_expiration 
    ON raw.databento_options_ohlcv_1d(expiration_date);
CREATE INDEX IF NOT EXISTS idx_options_strike 
    ON raw.databento_options_ohlcv_1d(strike_price);

-- Core symbols with active options markets on CME/CBOT/NYMEX:
-- Agricultural: ZL, ZS, ZM, ZC, ZW
-- Energy: CL, NG, HO, RB
-- Metals: GC, SI, HG
-- Rates: ZN, ZB, ZF
-- Equity: ES, NQ
-- FX: 6E, 6J, 6B


