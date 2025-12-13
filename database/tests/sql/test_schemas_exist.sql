-- Test: All required schemas exist
-- Returns 1 if all schemas exist, 0 otherwise

WITH required_schemas AS (
    SELECT unnest(['raw', 'staging', 'features', 'features_dev', 
                   'training', 'forecasts', 'reference', 'ops', 'explanations']) AS schema_name
),
existing_schemas AS (
    SELECT schema_name FROM information_schema.schemata
)
SELECT 
    CASE 
        WHEN COUNT(DISTINCT r.schema_name) = 9 THEN 1 
        ELSE 0 
    END AS test_passed,
    COUNT(DISTINCT r.schema_name) AS schemas_found,
    9 AS schemas_expected
FROM required_schemas r
JOIN existing_schemas e ON r.schema_name = e.schema_name;

