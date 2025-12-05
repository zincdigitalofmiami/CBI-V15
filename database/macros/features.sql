-- Feature Engineering Macros
-- Ported from daily_ml_matrix.sqlx

-- Price/Return/Volatility Block
-- Refactored to avoid nested window functions
CREATE OR REPLACE MACRO feat_price_block(sym) AS TABLE
WITH base_calcs AS (
    SELECT
        as_of_date,
        symbol,
        close,
        high,
        low,
        volume,
        -- Lags needed for returns
        LAG(close, 1) OVER (PARTITION BY symbol ORDER BY as_of_date) AS lag_close_1d,
        LAG(close, 2) OVER (PARTITION BY symbol ORDER BY as_of_date) AS lag_close_2d,
        LAG(close, 3) OVER (PARTITION BY symbol ORDER BY as_of_date) AS lag_close_3d,
        LAG(close, 5) OVER (PARTITION BY symbol ORDER BY as_of_date) AS lag_close_5d,
        LAG(close, 10) OVER (PARTITION BY symbol ORDER BY as_of_date) AS lag_close_10d,
        LAG(close, 21) OVER (PARTITION BY symbol ORDER BY as_of_date) AS lag_close_21d
    FROM raw.databento_ohlcv_daily
    WHERE symbol = sym
),
returns_calcs AS (
    SELECT
        *,
        -- Log Returns
        LN(close / NULLIF(lag_close_1d, 0)) AS log_ret_1d,
        LN(close / NULLIF(lag_close_2d, 0)) AS log_ret_2d,
        LN(close / NULLIF(lag_close_3d, 0)) AS log_ret_3d,
        LN(close / NULLIF(lag_close_5d, 0)) AS log_ret_5d,
        LN(close / NULLIF(lag_close_10d, 0)) AS log_ret_10d,
        LN(close / NULLIF(lag_close_21d, 0)) AS log_ret_21d
    FROM base_calcs
)
SELECT
    as_of_date,
    symbol,
    close,
    -- Lags
    lag_close_1d, lag_close_2d, lag_close_3d, lag_close_5d, lag_close_10d, lag_close_21d,
    -- Returns
    log_ret_1d, log_ret_2d, log_ret_3d, log_ret_5d, log_ret_10d, log_ret_21d,
    -- Moving Averages
    AVG(close) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS sma_5,
    AVG(close) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sma_10,
    AVG(close) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS sma_21,
    AVG(close) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,
    AVG(close) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS sma_200,
    -- Volatility (21d) - Now using pre-calculated log_ret_1d
    STDDEV(log_ret_1d) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) * SQRT(252) AS vol_21d,
    -- Volume Metrics
    AVG(volume) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS avg_volume_21d,
    volume / NULLIF(AVG(volume) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW), 0) AS volume_ratio,
    -- Range
    (high - low) / NULLIF(close, 0) AS daily_range_pct
FROM returns_calcs;

-- Targets Block (Future Returns)
CREATE OR REPLACE MACRO feat_targets_block(sym) AS TABLE
SELECT
    as_of_date,
    symbol,
    LEAD(close, 5)   OVER (PARTITION BY symbol ORDER BY as_of_date) AS target_price_1w,
    LEAD(close, 21)  OVER (PARTITION BY symbol ORDER BY as_of_date) AS target_price_1m,
    LEAD(close, 63)  OVER (PARTITION BY symbol ORDER BY as_of_date) AS target_price_3m,
    LEAD(close, 126) OVER (PARTITION BY symbol ORDER BY as_of_date) AS target_price_6m,
    -- Log Return Targets
    LN(LEAD(close, 5)   OVER (PARTITION BY symbol ORDER BY as_of_date) / NULLIF(close, 0)) AS target_ret_1w,
    LN(LEAD(close, 21)  OVER (PARTITION BY symbol ORDER BY as_of_date) / NULLIF(close, 0)) AS target_ret_1m,
    LN(LEAD(close, 63)  OVER (PARTITION BY symbol ORDER BY as_of_date) / NULLIF(close, 0)) AS target_ret_3m,
    LN(LEAD(close, 126) OVER (PARTITION BY symbol ORDER BY as_of_date) / NULLIF(close, 0)) AS target_ret_6m
FROM raw.databento_ohlcv_daily
WHERE symbol = sym;

-- Master Feature Matrix Macro
CREATE OR REPLACE MACRO feat_daily_ml_matrix_v15(sym) AS TABLE
WITH
    price_block AS (SELECT * FROM feat_price_block(sym)),
    target_block AS (SELECT * FROM feat_targets_block(sym))
SELECT
    p.as_of_date,
    p.symbol,
    -- Price Features
    p.close AS price_current,
    p.lag_close_1d, p.lag_close_2d, p.lag_close_3d, p.lag_close_5d, p.lag_close_10d, p.lag_close_21d,
    p.log_ret_1d, p.log_ret_2d, p.log_ret_3d, p.log_ret_5d, p.log_ret_10d, p.log_ret_21d,
    p.sma_5, p.sma_10, p.sma_21, p.sma_50, p.sma_200,
    -- MA Distances
    (p.close - p.sma_5) / NULLIF(p.sma_5, 0) AS dist_sma_5,
    (p.close - p.sma_10) / NULLIF(p.sma_10, 0) AS dist_sma_10,
    (p.close - p.sma_21) / NULLIF(p.sma_21, 0) AS dist_sma_21,
    (p.close - p.sma_50) / NULLIF(p.sma_50, 0) AS dist_sma_50,
    (p.close - p.sma_200) / NULLIF(p.sma_200, 0) AS dist_sma_200,
    p.vol_21d,
    p.avg_volume_21d,
    p.volume_ratio,
    p.daily_range_pct,
    -- Targets
    t.target_price_1w, t.target_price_1m, t.target_price_3m, t.target_price_6m,
    t.target_ret_1w, t.target_ret_1m, t.target_ret_3m, t.target_ret_6m
FROM price_block p
LEFT JOIN target_block t USING (as_of_date, symbol)
ORDER BY as_of_date;
