# Daily Forecast Pipeline

**Status:** Production  
**Last Updated:** December 3, 2025

## Schedule

**Daily at 5:00 PM (post-market close)**

## Steps

### 1. Data Ingestion
- Fetch latest ZL settlement price from Databento
- Update FRED macro indicators (if available)
- Ingest policy/news signals (ScrapeCreators, Trump feed)

### 2. Data Quality
- Run AnoFox EDA macros (gap detection, outlier detection)
- Apply prep macros (fill gaps, remove outliers)
- Log quality metrics to `data_quality_reports`

### 3. Feature Engineering
- Update all 80-120 production features
- Compute bucket-specific scores
- Update regime detection

### 4. Forecast Generation
- Run ensemble forecast (top 3-5 models per horizon)
- Generate 5 horizon forecasts (1W, 1M, 3M, 6M, 12M)
- Compute confidence intervals (90%, 95%)

### 5. Validation
- Compare forecast vs. actuals (previous day)
- Update model performance metrics
- Log to `validation_reports`

### 6. Export
- Write forecasts to `forecasts.forecast_reports`
- Update bucket scores in `sentiment_data.bucket_scores`
- Sync to Vercel API cache

## Monitoring

- Pipeline status logged to `ops.pipeline_runs`
- Alerts on failure via Vercel logs
- Model drift monitored in `/quant-admin`

## Rollback

If pipeline fails, previous day's forecast remains active until manual intervention.

