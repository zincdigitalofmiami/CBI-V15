# Views - Materialized and Logical Views

## Purpose
Additional views not tied to the API layer. For internal analysis and reporting.

## What Belongs Here
- `quant_report_view.sql` - Quant report generation
- Analysis views
- Debug views

## Relationship to 06_api
- `06_api/` = Views for dashboard consumption (public interface)
- `views/` = Views for internal/analyst use (internal tools)

## Note
Consider moving frequently-used views to `06_api/` if they need API exposure.

