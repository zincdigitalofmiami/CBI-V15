# GCP Cost Analysis for CBI-V15

**Date**: November 28, 2025  
**Project**: `cbi-v15`  
**Region**: `us-central1` (CRITICAL - multi-region costs 2-3x more!)

---

## 💰 Monthly Cost Breakdown

### 1. BigQuery (Primary Cost Driver)

#### Storage Costs
- **Active Storage**: $0.020 per GB/month
- **Long-term Storage** (90+ days): $0.010 per GB/month
- **Estimated Data Volume**: 
  - Raw data: ~50 GB (5 years historical)
  - Staging: ~30 GB
  - Features: ~20 GB
  - Training: ~10 GB
  - Forecasts: ~5 GB
  - **Total**: ~115 GB
- **Monthly Storage Cost**: ~$2.30/month

#### Query Costs
- **First 1 TB/month**: **FREE** ✅
- **After 1 TB**: $5.00 per TB
- **Estimated Monthly Scans**: 
  - Daily Dataform runs: ~50 GB/day × 30 = 1.5 TB/month
  - Training data exports: ~100 GB/month
  - Dashboard queries: ~50 GB/month
  - **Total**: ~1.65 TB/month
- **Query Cost**: (1.65 TB - 1 TB free) × $5 = **$3.25/month**

**BigQuery Total**: ~$5.55/month

---

### 2. Secret Manager

- **Cost**: $0.06 per secret per month
- **Secrets**: 4 (databento, fred, scrapecreators, glide)
- **Monthly Cost**: 4 × $0.06 = **$0.24/month**

---

### 3. Cloud Scheduler

- **First 3 jobs**: **FREE** ✅
- **After 3 jobs**: $0.10 per job per month
- **Estimated Jobs**: 
  - Daily ingestion: 1 job
  - Daily Dataform run: 1 job
  - Weekly model retrain: 1 job
  - **Total**: 3 jobs
- **Monthly Cost**: **$0.00/month** (within free tier)

---

### 4. Cloud Functions / Cloud Run

#### Cloud Functions (Serverless Ingestion)
- **Invocations**: First 2M/month **FREE** ✅
- **Estimated**: ~30 invocations/day × 30 = 900/month
- **Compute Time**: First 400,000 GB-seconds/month **FREE** ✅
- **Estimated**: ~5 seconds × 900 = 4,500 GB-seconds/month
- **Monthly Cost**: **$0.00/month** (within free tier)

#### Cloud Run (If Used)
- **CPU/Memory**: Pay per use, typically $0.00 for low usage
- **Estimated**: **$0.00/month** (if not used heavily)

---

### 5. Cloud Storage (GCS)

- **Standard Storage**: $0.020 per GB/month
- **Estimated**: ~10 GB (model artifacts, exports)
- **Monthly Cost**: ~$0.20/month

---

### 6. Dataform

- **Cost**: **FREE** ✅
- Dataform runs on BigQuery compute (already counted above)

---

### 7. Logging & Monitoring

#### Cloud Logging
- **Ingestion**: First 50 GB/month **FREE** ✅
- **Storage**: $0.50 per GB/month (after 30 days)
- **Estimated**: ~5 GB/month logs
- **Monthly Cost**: **$0.00/month** (within free tier)

#### Cloud Monitoring
- **Metrics**: First 150 MB/month **FREE** ✅
- **Estimated**: ~50 MB/month
- **Monthly Cost**: **$0.00/month** (within free tier)

---

### 8. Pub/Sub (Optional - Event-Driven)

- **Messages**: First 10 GB/month **FREE** ✅
- **Estimated**: ~1 GB/month (if used)
- **Monthly Cost**: **$0.00/month** (within free tier)

---

## 📊 Total Monthly Cost Estimate

| Service | Monthly Cost |
|---------|-------------|
| BigQuery Storage | $2.30 |
| BigQuery Queries | $3.25 |
| Secret Manager | $0.24 |
| Cloud Scheduler | $0.00 (free tier) |
| Cloud Functions | $0.00 (free tier) |
| Cloud Storage | $0.20 |
| Logging/Monitoring | $0.00 (free tier) |
| Dataform | $0.00 (free) |
| **TOTAL** | **~$6.00/month** |

---

## 💡 Cost Optimization Tips

### 1. BigQuery Query Optimization
- **Partition all tables by date** ✅ (already planned)
- **Cluster by symbol** ✅ (already planned)
- **Limit date ranges** in queries
- **Use materialized views** for frequent queries
- **Cache dashboard queries** (5-minute cache)

**Potential Savings**: Reduce query costs by 30-50% → **$2.00-2.50/month**

### 2. Storage Optimization
- **Use long-term storage** (90+ days) → 50% discount
- **Archive old data** to Cloud Storage (cheaper)
- **Compress Parquet exports** (snappy compression)

**Potential Savings**: Reduce storage by 40% → **$0.90/month**

### 3. Free Tier Maximization
- ✅ Cloud Scheduler: Stay within 3 jobs (free)
- ✅ Cloud Functions: Stay within 2M invocations (free)
- ✅ Logging: Stay within 50 GB/month (free)
- ✅ Monitoring: Stay within 150 MB/month (free)

---

## 📈 Cost Scaling Scenarios

### Low Usage (Current Estimate)
- **Monthly Cost**: ~$6.00/month
- **Annual Cost**: ~$72/year

### Medium Usage (2x data, 2x queries)
- **Monthly Cost**: ~$12.00/month
- **Annual Cost**: ~$144/year

### High Usage (5x data, 5x queries)
- **Monthly Cost**: ~$30.00/month
- **Annual Cost**: ~$360/year

---

## ⚠️ Cost Warnings

### 1. Multi-Region Costs
- **us-central1**: Current pricing
- **US multi-region**: 2-3x more expensive!
- **Other regions**: Varies, often more expensive
- **CRITICAL**: Always use `us-central1` only!

### 2. BigQuery Query Costs
- **Unpartitioned queries**: Can scan entire table (expensive!)
- **Full table scans**: Can cost $50+ per query
- **Solution**: Always partition by date, use date filters

### 3. Storage Growth
- **Historical data**: Grows linearly over time
- **5 years**: ~115 GB
- **10 years**: ~230 GB
- **Cost**: ~$4.60/month (10 years)

---

## 🎯 Cost Comparison: V14 vs V15

### V14 Issues (from `AI_MIGRATION_NIGHTMARE.md`)
- **Cloud SQL**: $139.87/month ❌
- **Storage movement**: ~$110/month ❌
- **Total**: ~$250/month ❌

### V15 (Optimized)
- **BigQuery only**: ~$6.00/month ✅
- **No Cloud SQL**: $0 ✅
- **No storage movement**: $0 ✅
- **Total**: ~$6.00/month ✅

**Savings**: ~$244/month (97% reduction!) 🎉

---

## 📋 Cost Monitoring

### Set Up Budget Alerts

```bash
# Create budget alert at $10/month
gcloud billing budgets create \
    --billing-account=YOUR_BILLING_ACCOUNT_ID \
    --display-name="CBI-V15 Budget" \
    --budget-amount=10USD \
    --threshold-rule=percent=80 \
    --threshold-rule=percent=100 \
    --project=cbi-v15
```

### Monitor Costs

1. **GCP Console**: https://console.cloud.google.com/billing
2. **BigQuery Usage**: https://console.cloud.google.com/bigquery/usage
3. **Cost Breakdown**: View by service, by project

---

## ✅ Cost Summary

**Estimated Monthly Cost**: **~$6.00/month**

**Breakdown**:
- BigQuery: ~$5.55/month (93% of total)
- Secret Manager: ~$0.24/month (4% of total)
- Cloud Storage: ~$0.20/month (3% of total)
- Everything else: FREE (within free tiers)

**Annual Cost**: **~$72/year**

**Compared to V14**: **97% cost reduction** ($250/month → $6/month)

---

## 🚨 Cost Guardrails

1. **Always use us-central1** (no multi-region!)
2. **Partition all tables by date** (reduce query costs)
3. **Set budget alerts** ($10/month threshold)
4. **Monitor BigQuery usage** (biggest cost driver)
5. **Archive old data** (use long-term storage discount)

---

**Last Updated**: November 28, 2025

