-- Raw Databento Futures OHLCV
-- Note: Databento API returns ts_event, collector normalizes to as_of_date
-- This is intentional for cross-source consistency (all date columns = as_of_date)

CREATE TABLE IF NOT EXISTS raw.databento_futures_ohlcv_1d (
    symbol VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,  -- Normalized from Databento ts_event
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    open_interest BIGINT,
    source VARCHAR DEFAULT 'databento',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, as_of_date)
);

-- 33 canonical symbols:
-- Agricultural: ZL, ZS, ZM, ZC, ZW, KE, ZO, CT, KC, SB, CC
-- Energy: CL, HO, RB, NG
-- Metals: GC, SI, HG, PA, PL
-- Treasuries: ZN, ZB, ZF
-- FX: 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S, DX
-- Palm: FCPO

CREATE INDEX IF NOT EXISTS idx_databento_ohlcv_symbol 
    ON raw.databento_futures_ohlcv_1d(symbol);
CREATE INDEX IF NOT EXISTS idx_databento_ohlcv_as_of_date 
    ON raw.databento_futures_ohlcv_1d(as_of_date);

