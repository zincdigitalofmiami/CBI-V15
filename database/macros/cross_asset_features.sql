-- ============================================================================
-- CROSS-ASSET CORRELATION & SPREAD FEATURES
-- ============================================================================
-- Build correlation matrices, betas, and spreads across all 30+ symbols
-- ============================================================================

-- ============================================================================
-- MACRO 1: Pairwise Correlation (Rolling)
-- ============================================================================
CREATE OR REPLACE MACRO calc_pairwise_correlation(sym1, sym2, lookback := 60) AS TABLE
WITH returns AS (
    SELECT
        a.as_of_date,
        a.symbol AS symbol1,
        b.symbol AS symbol2,
        LN(a.close / LAG(a.close, 1) OVER (PARTITION BY a.symbol ORDER BY a.as_of_date)) AS ret1,
        LN(b.close / LAG(b.close, 1) OVER (PARTITION BY b.symbol ORDER BY b.as_of_date)) AS ret2
    FROM raw.databento_ohlcv_daily a
    INNER JOIN raw.databento_ohlcv_daily b ON a.as_of_date = b.as_of_date
    WHERE a.symbol = sym1 AND b.symbol = sym2
)
SELECT
    as_of_date,
    symbol1,
    symbol2,
    CORR(ret1, ret2) OVER (
        ORDER BY as_of_date
        ROWS BETWEEN lookback - 1 PRECEDING AND CURRENT ROW
    ) AS corr_60d
FROM returns;

-- ============================================================================
-- MACRO 2: Rolling Beta (Asset vs Market/Benchmark)
-- ============================================================================
CREATE OR REPLACE MACRO calc_rolling_beta(sym, benchmark_sym, lookback := 60) AS TABLE
WITH returns AS (
    SELECT
        a.as_of_date,
        a.symbol,
        LN(a.close / LAG(a.close, 1) OVER (PARTITION BY a.symbol ORDER BY a.as_of_date)) AS ret_asset,
        LN(b.close / LAG(b.close, 1) OVER (PARTITION BY b.symbol ORDER BY b.as_of_date)) AS ret_benchmark
    FROM raw.databento_ohlcv_daily a
    INNER JOIN raw.databento_ohlcv_daily b ON a.as_of_date = b.as_of_date
    WHERE a.symbol = sym AND b.symbol = benchmark_sym
),
stats AS (
    SELECT
        as_of_date,
        symbol,
        ret_asset,
        ret_benchmark,
        COVAR_POP(ret_asset, ret_benchmark) OVER w AS covar,
        VAR_POP(ret_benchmark) OVER w AS var_benchmark
    FROM returns
    WINDOW w AS (ORDER BY as_of_date ROWS BETWEEN lookback - 1 PRECEDING AND CURRENT ROW)
)
SELECT
    as_of_date,
    symbol,
    covar / NULLIF(var_benchmark, 0) AS beta_60d
FROM stats;

-- ============================================================================
-- MACRO 3: Fundamental Spreads (Commodity-Specific)
-- ============================================================================
CREATE OR REPLACE MACRO calc_fundamental_spreads() AS TABLE
WITH prices AS (
    SELECT
        as_of_date,
        MAX(CASE WHEN symbol = 'ZL' THEN close END) AS zl_close,
        MAX(CASE WHEN symbol = 'ZS' THEN close END) AS zs_close,
        MAX(CASE WHEN symbol = 'ZM' THEN close END) AS zm_close,
        MAX(CASE WHEN symbol = 'CL' THEN close END) AS cl_close,
        MAX(CASE WHEN symbol = 'HO' THEN close END) AS ho_close,
        MAX(CASE WHEN symbol = 'RB' THEN close END) AS rb_close,
        MAX(CASE WHEN symbol = 'HG' THEN close END) AS hg_close,
        MAX(CASE WHEN symbol = 'GC' THEN close END) AS gc_close,
        MAX(CASE WHEN symbol = 'DX' THEN close END) AS dx_close
    FROM raw.databento_ohlcv_daily
    WHERE symbol IN ('ZL', 'ZS', 'ZM', 'CL', 'HO', 'RB', 'HG', 'GC', 'DX')
    GROUP BY as_of_date
)
SELECT
    as_of_date,
    -- Board Crush Spread: (ZM × 0.022 + ZL × 11) - ZS
    (zm_close * 0.022 + zl_close * 11) - zs_close AS board_crush_spread,
    -- Oil Share of Crush Value
    (zl_close * 11) / NULLIF((zm_close * 0.022 + zl_close * 11), 0) AS oil_share_of_crush,
    -- BOHO Spread: (ZL/100 × 7.5) - HO (Soy Oil vs Heating Oil)
    (zl_close / 100 * 7.5) - ho_close AS boho_spread,
    -- Crack Spread: (RB + HO) / 2 - CL (Refining margin proxy)
    ((rb_close + ho_close) / 2) - cl_close AS crack_spread,
    -- China Pulse: Copper as China demand proxy
    hg_close AS china_copper_proxy,
    -- Dollar Index (FX baseline)
    dx_close AS dollar_index
FROM prices;

-- ============================================================================
-- MACRO 4: Calendar Spreads (Near vs Far Month)
-- ============================================================================
-- Note: This requires multiple contract months in raw data
-- Placeholder for when we have ZLZ24, ZLH25, etc.
CREATE OR REPLACE MACRO calc_calendar_spreads(sym_near, sym_far) AS TABLE
WITH spreads AS (
    SELECT
        a.as_of_date,
        a.symbol AS near_contract,
        b.symbol AS far_contract,
        a.close AS near_price,
        b.close AS far_price,
        b.close - a.close AS calendar_spread_abs,
        (b.close - a.close) / NULLIF(a.close, 0) AS calendar_spread_pct
    FROM raw.databento_ohlcv_daily a
    INNER JOIN raw.databento_ohlcv_daily b ON a.as_of_date = b.as_of_date
    WHERE a.symbol = sym_near AND b.symbol = sym_far
)
SELECT * FROM spreads;

-- ============================================================================
-- MACRO 5: Cross-Asset Correlation Matrix (All Pairs)
-- ============================================================================
CREATE OR REPLACE MACRO calc_correlation_matrix(lookback := 60) AS TABLE
WITH symbols AS (
    SELECT DISTINCT symbol
    FROM raw.databento_ohlcv_daily
    WHERE symbol IN ('ZL', 'ZS', 'ZM', 'ZC', 'ZW', 'CL', 'HO', 'RB', 'NG',
                     'HG', 'GC', 'SI', 'PL', 'ZF', 'ZN', 'ZB', 'DX')
),
returns AS (
    SELECT
        as_of_date,
        symbol,
        LN(close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY as_of_date)) AS log_return
    FROM raw.databento_ohlcv_daily
    WHERE symbol IN (SELECT symbol FROM symbols)
),
pivoted AS (
    SELECT
        as_of_date,
        MAX(CASE WHEN symbol = 'ZL' THEN log_return END) AS ret_zl,
        MAX(CASE WHEN symbol = 'ZS' THEN log_return END) AS ret_zs,
        MAX(CASE WHEN symbol = 'ZM' THEN log_return END) AS ret_zm,
        MAX(CASE WHEN symbol = 'CL' THEN log_return END) AS ret_cl,
        MAX(CASE WHEN symbol = 'HO' THEN log_return END) AS ret_ho,
        MAX(CASE WHEN symbol = 'RB' THEN log_return END) AS ret_rb,
        MAX(CASE WHEN symbol = 'HG' THEN log_return END) AS ret_hg,
        MAX(CASE WHEN symbol = 'GC' THEN log_return END) AS ret_gc,
        MAX(CASE WHEN symbol = 'DX' THEN log_return END) AS ret_dx
    FROM returns
    GROUP BY as_of_date
)
SELECT
    as_of_date,
    -- ZL correlations
    CORR(ret_zl, ret_zs) OVER w AS corr_zl_zs_60d,
    CORR(ret_zl, ret_zm) OVER w AS corr_zl_zm_60d,
    CORR(ret_zl, ret_cl) OVER w AS corr_zl_cl_60d,
    CORR(ret_zl, ret_ho) OVER w AS corr_zl_ho_60d,
    CORR(ret_zl, ret_hg) OVER w AS corr_zl_hg_60d,
    CORR(ret_zl, ret_dx) OVER w AS corr_zl_dx_60d,
    -- CL correlations
    CORR(ret_cl, ret_ho) OVER w AS corr_cl_ho_60d,
    CORR(ret_cl, ret_rb) OVER w AS corr_cl_rb_60d,
    CORR(ret_cl, ret_dx) OVER w AS corr_cl_dx_60d,
    -- Metals correlations
    CORR(ret_hg, ret_gc) OVER w AS corr_hg_gc_60d,
    CORR(ret_hg, ret_dx) OVER w AS corr_hg_dx_60d
FROM pivoted
WINDOW w AS (ORDER BY as_of_date ROWS BETWEEN lookback - 1 PRECEDING AND CURRENT ROW);

