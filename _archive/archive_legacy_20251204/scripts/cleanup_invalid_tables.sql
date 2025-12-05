-- Cleanup invalid tables found in BigQuery audit
-- Run with: bq query --use_legacy_sql=false < scripts/cleanup_invalid_tables.sql

-- Remove invalid table names
DROP TABLE IF EXISTS `cbi-v15.raw.-----------------------------`;
DROP TABLE IF EXISTS `cbi-v15.staging.-----------------------------`;

-- Remove test table
DROP TABLE IF EXISTS `cbi-v15.raw.test_table`;

-- Verify cleanup
SELECT 
  'raw' as dataset,
  table_name
FROM `cbi-v15.raw.INFORMATION_SCHEMA.TABLES`
WHERE table_name LIKE '%-%' OR table_name = 'test_table'
UNION ALL
SELECT 
  'staging' as dataset,
  table_name
FROM `cbi-v15.staging.INFORMATION_SCHEMA.TABLES`
WHERE table_name LIKE '%-%';






