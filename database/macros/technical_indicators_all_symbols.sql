-- ============================================================================
-- TECHNICAL INDICATORS FOR ALL 31 FUTURES SYMBOLS
-- ============================================================================
-- Comprehensive technical analysis features for:
-- - Agricultural: ZL, ZS, ZM, ZC, ZW, ZO, HE, FCPO (8 symbols)
-- - Energy: CL, HO, RB, NG (4 symbols)
-- - Metals: HG, GC, SI, PL, PA (5 symbols)
-- - Treasuries: ZF, ZN, ZB (3 symbols)
-- - FX Futures: 6E, 6J, 6B, 6C, 6A, 6N, 6M, 6L, 6S, DX (10 symbols)
--
-- Total: 31 futures symbols
-- ============================================================================

-- ============================================================================
-- MACRO 1: RSI (Relative Strength Index)
-- ============================================================================
CREATE OR REPLACE MACRO calc_rsi(sym, period := 14) AS TABLE
WITH price_changes AS (
    SELECT
        as_of_date,
        symbol,
        close,
        close - LAG(close, 1) OVER w AS price_change
    FROM raw.databento_ohlcv_daily
    WHERE symbol = sym
    WINDOW w AS (PARTITION BY symbol ORDER BY as_of_date)
),
gains_losses AS (
    SELECT
        as_of_date,
        symbol,
        close,
        CASE WHEN price_change > 0 THEN price_change ELSE 0 END AS gain,
        CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END AS loss
    FROM price_changes
),
avg_gains_losses AS (
    SELECT
        as_of_date,
        symbol,
        close,
        AVG(gain) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN period - 1 PRECEDING AND CURRENT ROW) AS avg_gain,
        AVG(loss) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN period - 1 PRECEDING AND CURRENT ROW) AS avg_loss
    FROM gains_losses
)
SELECT
    as_of_date,
    symbol,
    close,
    CASE 
        WHEN avg_loss = 0 THEN 100
        ELSE 100 - (100 / (1 + (avg_gain / NULLIF(avg_loss, 0))))
    END AS rsi_14
FROM avg_gains_losses;

-- ============================================================================
-- MACRO 2: MACD (Moving Average Convergence Divergence)
-- ============================================================================
CREATE OR REPLACE MACRO calc_macd(sym, fast := 12, slow := 26, signal := 9) AS TABLE
WITH ema_calcs AS (
    SELECT
        as_of_date,
        symbol,
        close,
        -- EMA approximation using SMA (good enough for features)
        AVG(close) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN fast - 1 PRECEDING AND CURRENT ROW) AS ema_fast,
        AVG(close) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN slow - 1 PRECEDING AND CURRENT ROW) AS ema_slow
    FROM raw.databento_ohlcv_daily
    WHERE symbol = sym
),
macd_line AS (
    SELECT
        as_of_date,
        symbol,
        close,
        ema_fast - ema_slow AS macd
    FROM ema_calcs
)
SELECT
    as_of_date,
    symbol,
    close,
    macd,
    AVG(macd) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN signal - 1 PRECEDING AND CURRENT ROW) AS macd_signal,
    macd - AVG(macd) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN signal - 1 PRECEDING AND CURRENT ROW) AS macd_histogram
FROM macd_line;

-- ============================================================================
-- MACRO 3: Bollinger Bands
-- ============================================================================
CREATE OR REPLACE MACRO calc_bollinger(sym, period := 20, num_std := 2) AS TABLE
WITH stats AS (
    SELECT
        as_of_date,
        symbol,
        close,
        AVG(close) OVER w AS bb_middle,
        STDDEV(close) OVER w AS bb_std
    FROM raw.databento_ohlcv_daily
    WHERE symbol = sym
    WINDOW w AS (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN period - 1 PRECEDING AND CURRENT ROW)
)
SELECT
    as_of_date,
    symbol,
    close,
    bb_middle,
    bb_middle + (num_std * bb_std) AS bb_upper,
    bb_middle - (num_std * bb_std) AS bb_lower,
    (close - bb_middle) / NULLIF(bb_std, 0) AS bb_position,  -- Z-score
    (bb_upper - bb_lower) / NULLIF(bb_middle, 0) AS bb_width_pct
FROM stats;

-- ============================================================================
-- MACRO 4: ATR (Average True Range)
-- ============================================================================
CREATE OR REPLACE MACRO calc_atr(sym, period := 14) AS TABLE
WITH true_range AS (
    SELECT
        as_of_date,
        symbol,
        close,
        high,
        low,
        LAG(close, 1) OVER w AS prev_close,
        GREATEST(
            high - low,
            ABS(high - LAG(close, 1) OVER w),
            ABS(low - LAG(close, 1) OVER w)
        ) AS tr
    FROM raw.databento_ohlcv_daily
    WHERE symbol = sym
    WINDOW w AS (PARTITION BY symbol ORDER BY as_of_date)
)
SELECT
    as_of_date,
    symbol,
    close,
    AVG(tr) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN period - 1 PRECEDING AND CURRENT ROW) AS atr_14,
    tr / NULLIF(close, 0) AS tr_pct
FROM true_range;

-- ============================================================================
-- MACRO 5: Stochastic Oscillator
-- ============================================================================
CREATE OR REPLACE MACRO calc_stochastic(sym, period := 14, smooth := 3) AS TABLE
WITH high_low AS (
    SELECT
        as_of_date,
        symbol,
        close,
        high,
        low,
        MAX(high) OVER w AS highest_high,
        MIN(low) OVER w AS lowest_low
    FROM raw.databento_ohlcv_daily
    WHERE symbol = sym
    WINDOW w AS (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN period - 1 PRECEDING AND CURRENT ROW)
),
k_line AS (
    SELECT
        as_of_date,
        symbol,
        close,
        100 * (close - lowest_low) / NULLIF(highest_high - lowest_low, 0) AS stoch_k
    FROM high_low
)
SELECT
    as_of_date,
    symbol,
    close,
    stoch_k,
    AVG(stoch_k) OVER (PARTITION BY symbol ORDER BY as_of_date ROWS BETWEEN smooth - 1 PRECEDING AND CURRENT ROW) AS stoch_d
FROM k_line;

-- ============================================================================
-- MACRO 6: Momentum Indicators
-- ============================================================================
CREATE OR REPLACE MACRO calc_momentum(sym) AS TABLE
WITH price_data AS (
    SELECT
        as_of_date,
        symbol,
        close,
        LAG(close, 1) OVER w AS lag_1d,
        LAG(close, 5) OVER w AS lag_5d,
        LAG(close, 10) OVER w AS lag_10d,
        LAG(close, 21) OVER w AS lag_21d,
        LAG(close, 63) OVER w AS lag_63d
    FROM raw.databento_ohlcv_daily
    WHERE symbol = sym
    WINDOW w AS (PARTITION BY symbol ORDER BY as_of_date)
)
SELECT
    as_of_date,
    symbol,
    close,
    -- Rate of Change (ROC)
    (close - lag_10d) / NULLIF(lag_10d, 0) AS roc_10d,
    (close - lag_21d) / NULLIF(lag_21d, 0) AS roc_21d,
    (close - lag_63d) / NULLIF(lag_63d, 0) AS roc_63d,
    -- Momentum (absolute change)
    close - lag_10d AS momentum_10d,
    close - lag_21d AS momentum_21d
FROM price_data;

-- ============================================================================
-- MACRO 7: Volume Indicators
-- ============================================================================
CREATE OR REPLACE MACRO calc_volume_indicators(sym) AS TABLE
WITH volume_data AS (
    SELECT
        as_of_date,
        symbol,
        close,
        volume,
        LAG(close, 1) OVER w AS prev_close,
        AVG(volume) OVER (w ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS avg_volume_21d,
        STDDEV(volume) OVER (w ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS std_volume_21d
    FROM raw.databento_ohlcv_daily
    WHERE symbol = sym
    WINDOW w AS (PARTITION BY symbol ORDER BY as_of_date)
)
SELECT
    as_of_date,
    symbol,
    close,
    volume,
    avg_volume_21d,
    volume / NULLIF(avg_volume_21d, 0) AS volume_ratio,
    (volume - avg_volume_21d) / NULLIF(std_volume_21d, 0) AS volume_zscore,
    -- On-Balance Volume (OBV) approximation
    SUM(CASE
        WHEN close > prev_close THEN volume
        WHEN close < prev_close THEN -volume
        ELSE 0
    END) OVER (PARTITION BY symbol ORDER BY as_of_date) AS obv
FROM volume_data;

-- ============================================================================
-- MACRO 8: ALL TECHNICAL INDICATORS (Master Macro)
-- ============================================================================
CREATE OR REPLACE MACRO calc_all_technical_indicators(sym) AS TABLE
WITH
    rsi_data AS (SELECT * FROM calc_rsi(sym, 14)),
    macd_data AS (SELECT * FROM calc_macd(sym, 12, 26, 9)),
    bb_data AS (SELECT * FROM calc_bollinger(sym, 20, 2)),
    atr_data AS (SELECT * FROM calc_atr(sym, 14)),
    stoch_data AS (SELECT * FROM calc_stochastic(sym, 14, 3)),
    momentum_data AS (SELECT * FROM calc_momentum(sym)),
    volume_data AS (SELECT * FROM calc_volume_indicators(sym)),
    price_base AS (SELECT * FROM feat_price_block(sym))
SELECT
    p.as_of_date,
    p.symbol,
    -- Price & Basic Features
    p.close,
    p.lag_close_1d, p.lag_close_5d, p.lag_close_21d,
    p.log_ret_1d, p.log_ret_5d, p.log_ret_21d,
    p.sma_5, p.sma_10, p.sma_21, p.sma_50, p.sma_200,
    p.volatility_21d,
    -- RSI
    r.rsi_14,
    -- MACD
    m.macd, m.macd_signal, m.macd_histogram,
    -- Bollinger Bands
    b.bb_upper, b.bb_middle, b.bb_lower, b.bb_position, b.bb_width_pct,
    -- ATR
    a.atr_14, a.tr_pct,
    -- Stochastic
    s.stoch_k, s.stoch_d,
    -- Momentum
    mo.roc_10d, mo.roc_21d, mo.roc_63d, mo.momentum_10d, mo.momentum_21d,
    -- Volume
    v.volume, v.avg_volume_21d, v.volume_ratio, v.volume_zscore, v.obv
FROM price_base p
LEFT JOIN rsi_data r USING (as_of_date, symbol)
LEFT JOIN macd_data m USING (as_of_date, symbol)
LEFT JOIN bb_data b USING (as_of_date, symbol)
LEFT JOIN atr_data a USING (as_of_date, symbol)
LEFT JOIN stoch_data s USING (as_of_date, symbol)
LEFT JOIN momentum_data mo USING (as_of_date, symbol)
LEFT JOIN volume_data v USING (as_of_date, symbol)
ORDER BY as_of_date;

