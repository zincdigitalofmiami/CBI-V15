-- As-Of Join Helpers
-- Point-in-time safety for feature engineering
-- Prevents look-ahead bias in training data

-- Get the latest value as of a given date (forward-fill)
-- Usage: SELECT asof_value(raw.fred_economic, 'value', 'date', '2024-01-15', 'DXY')
CREATE OR REPLACE MACRO asof_latest(
    table_name, 
    value_col, 
    date_col, 
    as_of_date, 
    filter_value
) AS (
    SELECT value_col
    FROM table_name
    WHERE date_col <= as_of_date
    ORDER BY date_col DESC
    LIMIT 1
);

-- Safe lag that respects publication delays
-- For data with known publication lag (e.g., CFTC reports publish Tuesday for prior Friday)
CREATE OR REPLACE MACRO safe_lag(
    value,
    publication_lag_days
) AS (
    LAG(value, publication_lag_days) OVER (ORDER BY date)
);

-- Point-in-time join template for features
-- Ensures no data leakage when joining tables with different frequencies
-- 
-- Example: Joining daily prices with weekly CFTC data
-- The CFTC report for week ending Friday is published on Tuesday
-- So for Monday's feature row, we should use LAST week's CFTC data
--
-- SELECT 
--     p.date,
--     p.close,
--     asof_cftc.managed_money_net
-- FROM staging.ohlcv_daily p
-- ASOF JOIN staging.cftc_normalized asof_cftc
--     ON p.date >= asof_cftc.date + INTERVAL '4 days'  -- Tuesday publication
--     AND p.symbol = asof_cftc.commodity

-- Forward-fill macro for sparse data
-- Fills NULLs with the most recent non-NULL value
CREATE OR REPLACE MACRO forward_fill(
    value_col
) AS (
    COALESCE(
        value_col,
        LAST_VALUE(value_col IGNORE NULLS) OVER (
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
    )
);

-- Backward-looking window (safe for features)
CREATE OR REPLACE MACRO safe_rolling_avg(
    value_col,
    window_days
) AS (
    AVG(value_col) OVER (
        ORDER BY date
        ROWS BETWEEN window_days PRECEDING AND CURRENT ROW
    )
);

-- CRITICAL: Never use ROWS BETWEEN CURRENT ROW AND N FOLLOWING
-- That would leak future data into features!

