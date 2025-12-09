# 05_assertions - Data Quality Checks

## Purpose
SQL assertions for data quality - freshness checks, uniqueness, not-null constraints.

## What Belongs Here
- `assert_freshness.sql` - Check data is recent
- `assert_unique_keys.sql` - Check primary key uniqueness
- `assert_not_null.sql` - Check required fields
- `assert_valid_ranges.sql` - Check value bounds

## Inspired By
Dataform's assertion pattern - SQL that returns rows only when assertions FAIL.

## Pattern
```sql
-- assert_freshness.sql
-- Returns rows if data is stale (assertion fails)
SELECT 'raw.databento_ohlcv_daily' as table_name,
       MAX(trade_date) as last_date,
       CURRENT_DATE - MAX(trade_date) as days_stale
FROM raw.databento_ohlcv_daily
WHERE CURRENT_DATE - MAX(trade_date) > 2  -- Stale if > 2 days old
```

## Usage
Run assertions after ingestion to verify data quality.
Empty result = all assertions pass.

