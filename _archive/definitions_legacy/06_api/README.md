# API Views: Public Dashboard Views

Public-facing views for Next.js/Vercel dashboard.

## Purpose

API views expose forecast data and signals for dashboard consumption.

## Views

- `vw_latest_forecast.sqlx` - Latest forecasts by horizon
- `vw_big_eight_signals.sqlx` - Current Big 8 driver values
- `vw_regime_status.sqlx` - Current regime classification

## Security

Views are read-only and scoped to dashboard service account.

---

**Last Updated**: November 28, 2025

