-- Simple freshness test for forecasts.zl_predictions
-- Returns a single row with test_result and latest date

WITH latest AS (
  SELECT MAX(as_of_date) AS max_date
  FROM forecasts.zl_predictions
)
SELECT 
  CASE 
    WHEN max_date IS NULL THEN 'FAIL'
    WHEN max_date >= CURRENT_DATE - INTERVAL 14 DAY THEN 'PASS'
    ELSE 'WARN'
  END AS test_result,
  max_date
FROM latest;
