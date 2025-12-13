-- Test: No future leakage in training targets
-- CRITICAL: Targets must be computed from FUTURE data relative to as_of_date

-- For each row in features.targets:
-- target_1w should be the return from as_of_date to as_of_date + 5 trading days
-- This means the target value should NOT be available until 5 days AFTER as_of_date

-- Check: No training row should have a target for a future date that hasn't happened yet
WITH training_check AS (
    SELECT 
        date,
        target_1w,
        -- The 1w target requires data from date + 5 days
        -- So if date + 5 > current_date, we shouldn't have a non-null target
        CASE 
            WHEN date + INTERVAL '7 days' > CURRENT_DATE AND target_1w IS NOT NULL 
            THEN 1 
            ELSE 0 
        END AS has_future_leak
    FROM features.targets
    WHERE date IS NOT NULL
)
SELECT 
    CASE 
        WHEN COUNT(*) = 0 THEN 'SKIP'  -- Table empty
        WHEN SUM(has_future_leak) = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS test_result,
    SUM(has_future_leak) AS leaky_rows,
    COUNT(*) AS total_rows
FROM training_check;

