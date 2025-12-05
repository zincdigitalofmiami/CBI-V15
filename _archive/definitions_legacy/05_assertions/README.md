# Assertions: Data Quality Gates

Automated data quality checks that block deployment if failures occur.

## Purpose

Assertions ensure data quality before training and deployment.

## Assertions

- `assert_not_null_keys.sqlx` - Key columns must not be null
- `assert_unique_keys.sqlx` - Unique key constraints
- `assert_freshness.sqlx` - Data must be fresh (< 7 days old)
- `assert_crush_margin_valid.sqlx` - Crush margin sanity bounds
- `assert_join_integrity.sqlx` - Join completeness checks
- `assert_big_eight_complete.sqlx` - All Big 8 drivers must be present
- `assert_feature_collinearity.sqlx` - No feature pairs with correlation > 0.85

## Usage

Run assertions:
```bash
cd dataform
dataform test
```

Assertions run automatically in CI/CD pipeline.

---

**Last Updated**: November 28, 2025

