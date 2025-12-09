-- ============================================================================
-- BIG 8 BUCKET COT ENHANCEMENTS
-- ============================================================================
-- This file contains helper macros to integrate CFTC COT data into Big 8 buckets
-- 
-- COT Data provides positioning/sentiment signals:
-- - Managed Money = Speculators (hedge funds, CTAs)
-- - Producer/Merchant = Commercial hedgers ("smart money")
-- - Net positions as % of open interest = Sentiment gauge
-- - Extreme positioning (>30% of OI) = Potential reversal signal
-- - Hedger vs Speculator divergence = Contrarian signal
-- ============================================================================

-- ============================================================================
-- MACRO: Get COT Positioning Signal
-- ============================================================================
-- Returns a signal (-1, 0, 1) based on COT positioning
-- -1 = Bearish (overcrowded long, or hedgers short + specs long)
--  0 = Neutral
--  1 = Bullish (overcrowded short, or hedgers long + specs short)
-- ============================================================================

CREATE OR REPLACE MACRO calc_cot_signal(
    spec_net_pct DOUBLE,
    comm_net_pct DOUBLE
) AS (
    CASE
        -- Extreme long positioning (>30% of OI) = potential reversal (bearish)
        WHEN spec_net_pct > 30 THEN -1
        -- Extreme short positioning (<-30% of OI) = potential reversal (bullish)
        WHEN spec_net_pct < -30 THEN 1
        -- Hedgers long + Specs short = bullish (smart money signal)
        WHEN comm_net_pct > 10 AND spec_net_pct < -10 THEN 1
        -- Hedgers short + Specs long = bearish (smart money signal)
        WHEN comm_net_pct < -10 AND spec_net_pct > 10 THEN -1
        -- Neutral
        ELSE 0
    END
);

-- ============================================================================
-- MACRO: Get Latest COT Data for Symbol
-- ============================================================================
-- Forward-fills weekly COT data to daily frequency
-- ============================================================================

CREATE OR REPLACE MACRO get_latest_cot_disagg(symbol_code TEXT) AS TABLE
SELECT
    d.as_of_date,
    COALESCE(
        cot.managed_money_net_pct_oi,
        LAST_VALUE(cot.managed_money_net_pct_oi IGNORE NULLS) OVER (ORDER BY d.as_of_date)
    ) AS spec_net_pct,
    COALESCE(
        cot.prod_merc_net_pct_oi,
        LAST_VALUE(cot.prod_merc_net_pct_oi IGNORE NULLS) OVER (ORDER BY d.as_of_date)
    ) AS comm_net_pct,
    COALESCE(
        cot.open_interest,
        LAST_VALUE(cot.open_interest IGNORE NULLS) OVER (ORDER BY d.as_of_date)
    ) AS open_interest
FROM (
    SELECT DISTINCT as_of_date 
    FROM raw.databento_ohlcv_daily 
    WHERE symbol = symbol_code
) d
LEFT JOIN raw.cftc_cot_disaggregated cot 
    ON d.as_of_date = cot.report_date 
    AND cot.symbol = symbol_code;

-- ============================================================================
-- MACRO: Get Latest COT Data for FX/Treasuries (TFF Report)
-- ============================================================================

CREATE OR REPLACE MACRO get_latest_cot_tff(symbol_code TEXT) AS TABLE
SELECT
    d.as_of_date,
    COALESCE(
        cot.lev_money_net_pct_oi,
        LAST_VALUE(cot.lev_money_net_pct_oi IGNORE NULLS) OVER (ORDER BY d.as_of_date)
    ) AS spec_net_pct,
    COALESCE(
        cot.dealer_net,
        LAST_VALUE(cot.dealer_net IGNORE NULLS) OVER (ORDER BY d.as_of_date)
    ) AS dealer_net,
    COALESCE(
        cot.asset_mgr_net_pct_oi,
        LAST_VALUE(cot.asset_mgr_net_pct_oi IGNORE NULLS) OVER (ORDER BY d.as_of_date)
    ) AS asset_mgr_net_pct,
    COALESCE(
        cot.open_interest,
        LAST_VALUE(cot.open_interest IGNORE NULLS) OVER (ORDER BY d.as_of_date)
    ) AS open_interest
FROM (
    SELECT DISTINCT as_of_date 
    FROM raw.databento_ohlcv_daily 
    WHERE symbol = symbol_code
) d
LEFT JOIN raw.cftc_cot_tff cot 
    ON d.as_of_date = cot.report_date 
    AND cot.symbol = symbol_code;

-- ============================================================================
-- MACRO: Calculate COT Momentum (Week-over-Week Change)
-- ============================================================================

CREATE OR REPLACE MACRO calc_cot_momentum(symbol_code TEXT) AS TABLE
WITH cot_data AS (
    SELECT
        report_date,
        managed_money_net_pct_oi,
        LAG(managed_money_net_pct_oi, 1) OVER (ORDER BY report_date) AS prev_week_net_pct
    FROM raw.cftc_cot_disaggregated
    WHERE symbol = symbol_code
)
SELECT
    report_date,
    managed_money_net_pct_oi - prev_week_net_pct AS cot_momentum_1w,
    CASE
        WHEN managed_money_net_pct_oi - prev_week_net_pct > 5 THEN 1  -- Strong buying
        WHEN managed_money_net_pct_oi - prev_week_net_pct < -5 THEN -1  -- Strong selling
        ELSE 0
    END AS cot_momentum_signal
FROM cot_data
WHERE prev_week_net_pct IS NOT NULL;

-- ============================================================================
-- MACRO: Calculate COT Extremes (Z-Score)
-- ============================================================================

CREATE OR REPLACE MACRO calc_cot_extremes(symbol_code TEXT, lookback_weeks INT) AS TABLE
WITH cot_stats AS (
    SELECT
        report_date,
        managed_money_net_pct_oi,
        AVG(managed_money_net_pct_oi) OVER (
            ORDER BY report_date 
            ROWS BETWEEN lookback_weeks PRECEDING AND CURRENT ROW
        ) AS mean_net_pct,
        STDDEV(managed_money_net_pct_oi) OVER (
            ORDER BY report_date 
            ROWS BETWEEN lookback_weeks PRECEDING AND CURRENT ROW
        ) AS std_net_pct
    FROM raw.cftc_cot_disaggregated
    WHERE symbol = symbol_code
)
SELECT
    report_date,
    managed_money_net_pct_oi,
    (managed_money_net_pct_oi - mean_net_pct) / NULLIF(std_net_pct, 0) AS cot_zscore,
    CASE
        WHEN (managed_money_net_pct_oi - mean_net_pct) / NULLIF(std_net_pct, 0) > 2 THEN -1  -- Extreme long = bearish
        WHEN (managed_money_net_pct_oi - mean_net_pct) / NULLIF(std_net_pct, 0) < -2 THEN 1  -- Extreme short = bullish
        ELSE 0
    END AS cot_extreme_signal
FROM cot_stats;

-- ============================================================================
-- Example Usage
-- ============================================================================

-- Get latest COT data for ZL (Soybean Oil)
-- SELECT * FROM get_latest_cot_disagg('ZL') ORDER BY as_of_date DESC LIMIT 10;

-- Calculate COT signal for ZL
-- SELECT 
--     as_of_date,
--     spec_net_pct,
--     comm_net_pct,
--     calc_cot_signal(spec_net_pct, comm_net_pct) AS cot_signal
-- FROM get_latest_cot_disagg('ZL')
-- ORDER BY as_of_date DESC
-- LIMIT 10;

-- Get COT momentum for Copper (HG)
-- SELECT * FROM calc_cot_momentum('HG') ORDER BY report_date DESC LIMIT 10;

-- Get COT extremes for Crude Oil (CL)
-- SELECT * FROM calc_cot_extremes('CL', 52) ORDER BY report_date DESC LIMIT 10;

