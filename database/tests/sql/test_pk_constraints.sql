-- Test: Primary key constraints are defined
-- Checks that key tables have PKs

WITH expected_pks AS (
    SELECT * FROM (VALUES
        ('raw', 'databento_futures_ohlcv_1d'),
        ('raw', 'fred_economic'),
        ('raw', 'cftc_cot'),
        ('staging', 'ohlcv_daily'),
        ('staging', 'market_daily'),
        ('features', 'daily_ml_matrix_zl'),
        ('features', 'targets'),
        ('training', 'bucket_predictions'),
        ('forecasts', 'zl_predictions'),
        ('ops', 'ingestion_completion')
    ) AS t(schema_name, table_name)
)
SELECT 
    e.schema_name,
    e.table_name,
    CASE WHEN tc.constraint_type IS NOT NULL THEN 'PASS' ELSE 'FAIL' END AS test_result
FROM expected_pks e
LEFT JOIN information_schema.table_constraints tc 
    ON tc.table_schema = e.schema_name 
    AND tc.table_name = e.table_name 
    AND tc.constraint_type = 'PRIMARY KEY'
ORDER BY e.schema_name, e.table_name;

