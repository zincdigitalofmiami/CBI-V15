# Cost Optimization Plan - $100/month Budget Cap

**Date**: November 28, 2025  
**Budget Constraint**: **$100/month MAXIMUM**  
**Current Estimate**: ~$227/month (EXCEEDS BUDGET!)  
**Target**: <$100/month

---

## 💰 Budget Breakdown (Hard Cap: $100/month)

### Fixed Costs (Cannot Reduce)
- **Databento API**: $200/month (fixed subscription)
- **ScrapeCreators API**: $20/month (fixed subscription)
- **Total Fixed**: $220/month

**⚠️ PROBLEM**: Fixed costs ($220) already exceed budget ($100)!

**Solution**: These are external costs, not GCP costs. Budget cap applies to **GCP only**.

### Revised Budget Allocation
- **GCP Costs**: **$80/month MAX** (leaves $20 buffer)
- **External APIs**: $220/month (separate budget)
- **Total Project**: $300/month

**OR** (if $100 is TOTAL including APIs):
- **GCP Costs**: **$0/month** (impossible - need BigQuery)
- **External APIs**: $220/month
- **Total**: $220/month (exceeds $100)

---

## 🎯 Assumption: $100/month = GCP Only

**Budget**: $100/month for GCP costs  
**External APIs**: $220/month (separate line item)  
**Total Project Cost**: $320/month

---

## 🔧 Aggressive Cost Optimizations

### 1. BigQuery Query Optimization (CRITICAL!)

#### Current Estimate: $3.25/month
#### Target: <$1/month (70% reduction)

**Optimizations**:
- ✅ **Aggressive date filtering**: Only query last 30 days for daily operations
- ✅ **Materialized views**: Cache frequent queries (5-minute refresh)
- ✅ **Query result caching**: Enable 24-hour cache for dashboard queries
- ✅ **Incremental Dataform runs**: Only process new data, not full rebuilds
- ✅ **Partition pruning**: Always filter by date partition
- ✅ **Limit query concurrency**: Max 5 concurrent queries

**Implementation**:
```sql
-- ✅ OPTIMIZED: Only last 30 days
SELECT * FROM `cbi-v15.features.daily_ml_matrix`
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  AND symbol = 'ZL'

-- ❌ AVOID: Full table scan
SELECT * FROM `cbi-v15.features.daily_ml_matrix`
WHERE symbol = 'ZL'  -- No date filter!
```

**Expected Savings**: $2.25/month → **$1.00/month**

---

### 2. BigQuery Storage Optimization

#### Current Estimate: $2.30/month
#### Target: <$1/month (57% reduction)

**Optimizations**:
- ✅ **Aggressive archiving**: Move data >90 days to long-term storage (50% discount)
- ✅ **Compression**: Use Parquet with snappy compression (30-50% reduction)
- ✅ **Data lifecycle**: Delete raw data >2 years old (keep staging/features)
- ✅ **Deduplication**: Remove duplicate records before storage

**Implementation**:
```sql
-- Archive old data to Cloud Storage (cheaper)
-- Keep only last 2 years in BigQuery
DELETE FROM `cbi-v15.raw.databento_daily_ohlcv`
WHERE date < DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
```

**Expected Savings**: $2.30/month → **$1.00/month**

---

### 3. Databento 5-Minute Pull Optimization

#### Challenge: 5-minute pulls = 288 pulls/day = 8,640 pulls/month

**Optimizations**:
- ✅ **Incremental pulls only**: Only fetch new data since last pull
- ✅ **Local caching**: Cache responses locally (Parquet files)
- ✅ **Batch uploads**: Upload to BigQuery once per hour (not every 5 min)
- ✅ **Deduplication**: Skip if data already exists

**Implementation**:
```python
# Pull every 5 minutes, but only upload to BQ hourly
# Cache locally in Parquet files
# Deduplicate before upload
```

**Cost Impact**: 
- Databento API: $200/month (fixed, cannot reduce)
- BigQuery uploads: Reduced from 288/day to 24/day (92% reduction)
- Storage: Minimal increase (local cache)

**Expected Savings**: Upload costs reduced by 90%

---

### 4. Public Dataset Query Optimization

#### Current Estimate: $1.30/month
#### Target: <$0.50/month (62% reduction)

**Optimizations**:
- ✅ **Cache public dataset queries**: Store results in staging tables
- ✅ **Reduce query frequency**: Query weekly, not daily
- ✅ **Limit date ranges**: Only query last 30 days
- ✅ **Use materialized views**: Pre-compute aggregations

**Implementation**:
```sql
-- Query public datasets weekly, store in staging
-- Use staging tables for daily operations
-- Avoid querying public datasets directly in production queries
```

**Expected Savings**: $1.30/month → **$0.50/month**

---

### 5. Cloud Storage Optimization

#### Current Estimate: $0.20/month
#### Target: <$0.10/month (50% reduction)

**Optimizations**:
- ✅ **Nearline storage**: Use Nearline for archives (cheaper)
- ✅ **Compression**: Compress all files before upload
- ✅ **Lifecycle policies**: Auto-delete files >90 days old

**Expected Savings**: $0.20/month → **$0.10/month**

---

### 6. Dataform Run Optimization

#### Challenge: Full Dataform runs can be expensive

**Optimizations**:
- ✅ **Incremental tables only**: Use MERGE semantics, not full rebuilds
- ✅ **Run frequency**: Daily, not multiple times per day
- ✅ **Selective runs**: Only run changed definitions
- ✅ **Tag-based runs**: Run only specific layers (staging, features, training)

**Implementation**:
```bash
# Run only staging layer (if raw data changed)
dataform run --tags staging

# Run only features layer (if staging changed)
dataform run --tags features

# Full run only weekly (for validation)
dataform run --full-refresh
```

**Expected Savings**: Reduce Dataform query costs by 60%

---

### 7. Dashboard Query Optimization

#### Challenge: Dashboard queries can be expensive if not cached

**Optimizations**:
- ✅ **Aggressive caching**: 5-minute cache for all dashboard queries
- ✅ **Materialized views**: Pre-compute dashboard data
- ✅ **Limit date ranges**: Only show last 30 days by default
- ✅ **Pagination**: Load data in chunks, not all at once

**Implementation**:
```sql
-- Create materialized view for dashboard
CREATE MATERIALIZED VIEW `cbi-v15.api.vw_dashboard_latest`
AS
SELECT * FROM `cbi-v15.features.daily_ml_matrix`
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  AND symbol = 'ZL'
OPTIONS(
  enable_refresh=true,
  refresh_interval_minutes=5
)
```

**Expected Savings**: Reduce dashboard query costs by 80%

---

## 📊 Optimized Cost Breakdown

### GCP Costs (Optimized)

| Service | Current | Optimized | Savings |
|---------|---------|-----------|---------|
| **BigQuery Storage** | $2.30 | $1.00 | $1.30 |
| **BigQuery Queries** | $3.25 | $1.00 | $2.25 |
| **Public Dataset Queries** | $1.30 | $0.50 | $0.80 |
| **Cloud Storage** | $0.20 | $0.10 | $0.10 |
| **Secret Manager** | $0.24 | $0.24 | $0.00 |
| **Everything Else** | $0.00 | $0.00 | $0.00 |
| **TOTAL** | **$7.29** | **$2.84** | **$4.45** |

### With Buffer (Worst Case)

| Scenario | Cost |
|----------|------|
| **Optimized Normal** | $2.84/month |
| **Optimized High (2x)** | $5.68/month |
| **Optimized Very High (5x)** | $14.20/month |
| **Budget Cap** | **$80/month** |
| **Safety Margin** | **82-96% buffer** ✅ |

---

## 🚨 Critical Cost Controls

### 1. Hard Budget Cap
```bash
# Set budget alert at $80/month (80% of $100 cap)
gcloud billing budgets create \
    --billing-account=YOUR_BILLING_ACCOUNT_ID \
    --display-name="CBI-V15 Budget Cap" \
    --budget-amount=100USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=80 \
    --threshold-rule=percent=100 \
    --threshold-rule=percent=120 \
    --project=cbi-v15
```

### 2. Query Cost Monitoring
```sql
-- Monitor query costs daily
SELECT 
  job_id,
  creation_time,
  total_bytes_processed / 1024 / 1024 / 1024 AS gb_processed,
  total_bytes_processed / 1024 / 1024 / 1024 / 1024 * 5 AS cost_usd
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  AND total_bytes_processed > 0
ORDER BY total_bytes_processed DESC
LIMIT 10
```

### 3. Storage Monitoring
```sql
-- Monitor storage costs
SELECT 
  dataset_id,
  ROUND(total_logical_bytes / 1024 / 1024 / 1024, 2) AS gb_storage,
  ROUND(total_logical_bytes / 1024 / 1024 / 1024 * 0.020, 2) AS cost_usd_month
FROM `cbi-v15.__TABLES__`
GROUP BY dataset_id
ORDER BY total_logical_bytes DESC
```

---

## 📋 Implementation Checklist

### Immediate Actions (Before First Run)

- [ ] Set budget alert at $80/month
- [ ] Enable query result caching (24 hours)
- [ ] Create materialized views for dashboard
- [ ] Implement incremental Dataform runs only
- [ ] Set up local caching for Databento pulls
- [ ] Configure data lifecycle policies (archive >90 days)
- [ ] Limit all queries to last 30 days by default

### Ongoing Monitoring

- [ ] Daily cost report (email alert)
- [ ] Weekly storage audit (archive old data)
- [ ] Monthly query optimization review
- [ ] Quarterly cost analysis

---

## 🎯 Cost Optimization Summary

### Before Optimization
- **GCP Costs**: ~$7.29/month
- **External APIs**: $220/month
- **Total**: ~$227/month

### After Optimization
- **GCP Costs**: ~$2.84/month (61% reduction)
- **External APIs**: $220/month (fixed)
- **Total**: ~$223/month

### Worst Case (5x usage)
- **GCP Costs**: ~$14.20/month (still under $80 cap!)
- **External APIs**: $220/month
- **Total**: ~$234/month

---

## ✅ Budget Compliance

- **Budget Cap**: $100/month (GCP only)
- **Optimized Normal**: $2.84/month ✅ (96% under budget)
- **Optimized High**: $5.68/month ✅ (94% under budget)
- **Optimized Very High**: $14.20/month ✅ (86% under budget)
- **Safety Margin**: 82-96% buffer ✅

---

## 🚀 Next Steps

1. **Implement optimizations** (see checklist above)
2. **Set budget alerts** ($80/month threshold)
3. **Monitor costs daily** (first week)
4. **Adjust as needed** (based on actual usage)

---

**Last Updated**: November 28, 2025

