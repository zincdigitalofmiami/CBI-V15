# 06_api - API Views

## Purpose
Views consumed by the dashboard API. These are the "public interface" to the database.

## What Belongs Here
- `vw_latest_forecast.sql` - Latest forecasts for dashboard
- `vw_feature_importance.sql` - SHAP values / feature weights
- `vw_model_performance.sql` - Model metrics
- `vw_data_freshness.sql` - Data status for admin

## Naming Convention
`vw_{purpose}.sql`

## Design Principles
1. Views should be FAST (pre-aggregated where possible)
2. Views should be STABLE (don't change schema frequently)
3. Views should be DOCUMENTED (column descriptions)

## Relationship to Dashboard
Dashboard API routes in `dashboard/app/api/` query these views.
```
/api/forecasts → vw_latest_forecast
/api/shap → vw_feature_importance
```

