# Data Quality Gates

**Status:** Production  
**Last Updated:** December 3, 2025

## AnoFox EDA & Prep Procedures

All raw data must pass quality gates before entering the forecasting pipeline.

## Gate 1: Gap Detection

**Tool:** `TS_DETECT_GAPS()` macro

**Thresholds:**
- Max gap size: 5 trading days
- Gaps > 5 days: FAIL (manual review required)

**Action on FAIL:** Fill gaps using linear interpolation, log warning

## Gate 2: Outlier Detection

**Tool:** `TS_OUTLIER_DETECT()` macro

**Method:** Z-score (3σ threshold)

**Thresholds:**
- Outlier ratio ≤ 2%: PASS
- Outlier ratio 2-5%: WARNING
- Outlier ratio > 5%: FAIL (manual review required)

**Action on WARNING:** Winsorize at 3σ, log to `data_quality_reports`

**Action on FAIL:** Quarantine data, alert operator

## Gate 3: Null Ratio

**Threshold:**
- Null ratio ≤ 1%: PASS
- Null ratio 1-5%: WARNING
- Null ratio > 5%: FAIL

**Action on WARNING:** Forward fill if time-series, otherwise drop rows

**Action on FAIL:** Reject data source, alert operator

## Gate 4: Stationarity

**Tool:** AnoFox trend strength analysis

**Thresholds:**
- Trend strength < 0.8: PASS (stationary enough)
- Trend strength ≥ 0.8: Apply differencing

**Action:** Log trend strength to metadata

## Gate 5: Seasonality Detection

**Tool:** AnoFox automatic seasonality detection

**Expected:** 252 trading days (annual crop cycle)

**Action:** If detected period ≠ 252, log for review (not auto-fail)

## Monitoring

All gate results logged to `data_quality_reports` table daily.

Dashboard: `/quant-reports` → Data Quality Gates section

