# SQL Smoke Testing Guide

This guide describes how to run quick SQL smoke tests against MotherDuck (read-only) or local DuckDB to validate that critical tables are present and populated.

## Files

- `database/tests/harness.py` — Python runner for `.sql` tests under `database/tests/sql/`
- `scripts/sql_smoke_tests.py` — Convenience script to run a small invariant set

## Running locally

```
export MOTHERDUCK_DB=cbi_v15
# export MOTHERDUCK_TOKEN=mdp_xxx  # optional for cloud

python scripts/sql_smoke_tests.py --motherduck   # cloud (read-only)
python scripts/sql_smoke_tests.py --local        # local duckdb
```

## Adding a new smoke test

Create a file under `database/tests/sql/test_<name>.sql` that returns a single row with a `test_result` column of `PASS`, `FAIL`, or `WARN` and any diagnostic columns.

Example:

```
-- Verify forecasts table has recent data
WITH latest AS (
  SELECT max(as_of_date) AS max_date
  FROM forecasts.zl_predictions
)
SELECT
  CASE WHEN max_date >= current_date - INTERVAL 7 DAY THEN 'PASS'
       ELSE 'FAIL'
  END AS test_result,
  max_date
FROM latest;
```

Keep tests fast and deterministic. Avoid heavy scans.
