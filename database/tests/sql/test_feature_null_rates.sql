-- Test: Feature null rates within acceptable bounds
-- High null rates indicate data quality issues

WITH null_counts AS (
    SELECT
        COUNT(*) AS total_rows,
        COUNT(databento_zl_close) AS non_null_close,
        COUNT(volatility_zl_21d) AS non_null_vol,
        COUNT(tech_zl_rsi_14) AS non_null_rsi,
        COUNT(fx_dxy) AS non_null_dxy
    FROM features.daily_ml_matrix_zl
)
SELECT 
    CASE 
        WHEN total_rows = 0 THEN 'SKIP'
        WHEN non_null_close::FLOAT / total_rows >= 0.95
             AND non_null_vol::FLOAT / total_rows >= 0.90
             AND non_null_rsi::FLOAT / total_rows >= 0.90
             AND non_null_dxy::FLOAT / total_rows >= 0.95
        THEN 'PASS'
        ELSE 'WARN'
    END AS test_result,
    total_rows,
    ROUND(100.0 * non_null_close / NULLIF(total_rows, 0), 2) AS pct_close,
    ROUND(100.0 * non_null_vol / NULLIF(total_rows, 0), 2) AS pct_vol,
    ROUND(100.0 * non_null_rsi / NULLIF(total_rows, 0), 2) AS pct_rsi,
    ROUND(100.0 * non_null_dxy / NULLIF(total_rows, 0), 2) AS pct_dxy
FROM null_counts;

