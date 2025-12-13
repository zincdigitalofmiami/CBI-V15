-- Test: Data freshness check
-- Verifies that key tables have recent data

WITH freshness AS (
    SELECT 'raw.databento_futures_ohlcv_1d' AS table_name, 
           MAX(date) AS max_date,
           CURRENT_DATE - MAX(date) AS days_stale
    FROM raw.databento_futures_ohlcv_1d
    UNION ALL
    SELECT 'raw.fred_economic', MAX(date), CURRENT_DATE - MAX(date)
    FROM raw.fred_economic
    UNION ALL
    SELECT 'staging.market_daily', MAX(date), CURRENT_DATE - MAX(date)
    FROM staging.market_daily
    UNION ALL
    SELECT 'features.daily_ml_matrix_zl', MAX(as_of_date), CURRENT_DATE - MAX(as_of_date)
    FROM features.daily_ml_matrix_zl
)
SELECT 
    table_name,
    max_date,
    days_stale,
    CASE 
        WHEN max_date IS NULL THEN 'SKIP'
        WHEN days_stale <= 3 THEN 'PASS'
        WHEN days_stale <= 7 THEN 'WARN'
        ELSE 'FAIL'
    END AS test_result
FROM freshness
ORDER BY table_name;

