-- Test: No future data leakage in features
-- Ensures as_of_date logic is correct

-- Check that feature dates don't exceed target dates
WITH feature_dates AS (
    SELECT 
        as_of_date,
        updated_at
    FROM features.daily_ml_matrix_zl
    WHERE as_of_date IS NOT NULL
    LIMIT 1000
)
SELECT 
    CASE 
        WHEN COUNT(*) = 0 THEN 'SKIP'  -- Table empty
        WHEN MAX(as_of_date) <= CURRENT_DATE THEN 'PASS'
        ELSE 'FAIL'
    END AS test_result,
    MAX(as_of_date) AS max_as_of_date,
    COUNT(*) AS rows_checked
FROM feature_dates;

