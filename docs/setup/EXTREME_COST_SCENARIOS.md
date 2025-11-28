# Extreme High Cost Scenarios - Data Pull & Scraping

**Date**: November 28, 2025  
**Purpose**: Analyze worst-case costs for heavy data ingestion, API calls, and scraping

---

## 🔥 Extreme High Scenario Analysis

### Scenario: Maximum Data Pull (All Sources, High Frequency)

---

## 1. API Call Costs

### Databento API
- **Cost Model**: Pay-per-use (not included in GCP)
- **Daily Calls**: 
  - Daily OHLCV: 6 symbols × 1 call = 6 calls/day
  - Intraday (5-min): 6 symbols × 288 intervals = 1,728 calls/day
  - Historical backfill: 1,000 calls (one-time)
- **Monthly Calls**: ~52,000 calls
- **Cost**: Varies by plan (typically $0.001-0.01 per call)
- **Extreme High**: **$520/month** (if $0.01/call)

### FRED API
- **Cost**: **FREE** ✅ (Federal Reserve Economic Data)
- **Rate Limit**: 120 calls/minute
- **Daily Calls**: 60 series × 1 call = 60 calls/day
- **Monthly Cost**: **$0.00/month**

### ScrapeCreators API
- **Cost Model**: Subscription-based (not GCP)
- **Monthly Subscription**: ~$50-200/month (varies by plan)
- **Extreme High**: **$200/month**

### USDA/EIA/CFTC APIs
- **Cost**: **FREE** ✅ (Government APIs)
- **Monthly Cost**: **$0.00/month**

### Weather APIs (NOAA, INMET, Argentina SMN)
- **NOAA**: **FREE** ✅
- **INMET**: **FREE** ✅
- **Argentina SMN**: **FREE** ✅
- **Monthly Cost**: **$0.00/month**

**API Call Costs Total**: **$720/month** (worst case)

---

## 2. BigQuery Public Dataset Query Costs

### Google Public Datasets (GSOD, GFS, GDELT, BLS, FEC)
- **Cost**: Standard BigQuery pricing ($5/TB after 1 TB free)
- **Daily Queries**:
  - GSOD weather: ~10 GB/day
  - GFS forecasts: ~5 GB/day
  - GDELT events: ~20 GB/day
  - BLS economic: ~2 GB/day
  - FEC contributions: ~5 GB/day
- **Monthly Scans**: ~1.26 TB/month
- **Cost**: (1.26 TB - 1 TB free) × $5 = **$1.30/month**

### Extreme High Scenario (10x queries)
- **Monthly Scans**: ~12.6 TB/month
- **Cost**: (12.6 TB - 1 TB free) × $5 = **$58.00/month**

---

## 3. Cloud Functions / Cloud Run (Scraping & Ingestion)

### Cloud Functions (Serverless Ingestion)
- **Invocations**: First 2M/month **FREE** ✅
- **Extreme High**: 
  - 100 invocations/day × 30 = 3,000/month
  - Still within free tier ✅
- **Compute Time**: First 400,000 GB-seconds/month **FREE** ✅
- **Extreme High**:
  - 10 seconds × 3,000 invocations × 1 GB = 30,000 GB-seconds/month
  - Still within free tier ✅
- **Monthly Cost**: **$0.00/month**

### Cloud Run (Heavy Scraping Jobs)
- **CPU/Memory**: Pay per use
- **Extreme High Scenario**:
  - 2 vCPU × 4 GB RAM × 1 hour/day × 30 days
  - Cost: ~$0.10/hour × 30 hours = **$3.00/month**

---

## 4. Cloud Storage (Scraped Data Storage)

### Storage Costs
- **Standard Storage**: $0.020 per GB/month
- **Extreme High Scenario**:
  - Scraped HTML/JSON: ~50 GB/month
  - Model artifacts: ~20 GB/month
  - Exports/backups: ~30 GB/month
  - **Total**: ~100 GB
- **Monthly Cost**: 100 GB × $0.020 = **$2.00/month**

### Network Egress (Downloading Scraped Data)
- **First 1 GB/month**: **FREE** ✅
- **After 1 GB**: $0.12 per GB
- **Extreme High**: 50 GB/month × $0.12 = **$6.00/month**

**Storage Total**: **$8.00/month**

---

## 5. BigQuery Storage (Ingested Data)

### Storage Growth (Extreme High)
- **Raw Data**: 
  - Current: ~50 GB
  - Extreme High: ~500 GB (10x historical data)
- **Staging**: ~300 GB
- **Features**: ~200 GB
- **Training**: ~100 GB
- **Forecasts**: ~50 GB
- **Total**: ~1,150 GB
- **Monthly Cost**: 1,150 GB × $0.020 = **$23.00/month**

### Long-Term Storage Discount (90+ days)
- **After 90 days**: 50% discount
- **Effective Cost**: ~$11.50/month (if using long-term storage)

---

## 6. BigQuery Query Costs (Heavy Usage)

### Normal Usage
- **Monthly Scans**: ~1.65 TB/month
- **Cost**: (1.65 TB - 1 TB free) × $5 = **$3.25/month**

### Extreme High Usage
- **Daily Dataform runs**: 200 GB/day × 30 = 6 TB/month
- **Training data exports**: 500 GB/month
- **Dashboard queries**: 200 GB/month
- **Backtesting queries**: 1 TB/month
- **Total**: ~7.7 TB/month
- **Cost**: (7.7 TB - 1 TB free) × $5 = **$33.50/month**

---

## 7. Cloud Scheduler (High Frequency Jobs)

### Normal Usage
- **3 jobs**: **FREE** ✅

### Extreme High Usage
- **20 jobs** (multiple data sources, high frequency):
  - First 3: FREE
  - Remaining 17: 17 × $0.10 = **$1.70/month**

---

## 8. Secret Manager (API Keys)

- **Cost**: $0.06 per secret per month
- **Secrets**: 4-10 (depending on sources)
- **Extreme High**: 10 × $0.06 = **$0.60/month**

---

## 📊 Extreme High Cost Summary

### Worst-Case Scenario (All Maximums)

| Service | Normal | Extreme High |
|---------|--------|--------------|
| **API Calls** (external) | $0 | **$720** |
| **BigQuery Storage** | $2.30 | **$23.00** |
| **BigQuery Queries** | $3.25 | **$33.50** |
| **Public Dataset Queries** | $1.30 | **$58.00** |
| **Cloud Storage** | $0.20 | **$2.00** |
| **Network Egress** | $0 | **$6.00** |
| **Cloud Run** | $0 | **$3.00** |
| **Cloud Scheduler** | $0 | **$1.70** |
| **Secret Manager** | $0.24 | **$0.60** |
| **GCP Subtotal** | **$7.29** | **$127.80** |
| **External APIs** | **$0** | **$720.00** |
| **TOTAL** | **~$7/month** | **~$848/month** |

---

## 🎯 Realistic High Scenario (More Likely)

### Assumptions:
- Databento: Standard plan (~$100/month)
- ScrapeCreators: Pro plan (~$100/month)
- 3x normal data volume
- 3x normal query volume

| Service | Cost |
|---------|------|
| **External APIs** | $200/month |
| **BigQuery Storage** | $7.00/month |
| **BigQuery Queries** | $10.00/month |
| **Public Dataset Queries** | $5.00/month |
| **Cloud Storage** | $1.00/month |
| **Network Egress** | $2.00/month |
| **Everything Else** | $1.00/month |
| **TOTAL** | **~$226/month** |

---

## ⚠️ Cost Risk Factors

### 1. Unoptimized Queries (Biggest Risk!)
- **Full table scans**: Can cost $50+ per query
- **Unpartitioned tables**: Scans entire history
- **No date filters**: Scans all data
- **Risk**: **$100-500/month** if not careful

### 2. Multi-Region Storage
- **us-central1**: Current pricing
- **US multi-region**: 2-3x more expensive
- **Risk**: **$200-400/month** if misconfigured

### 3. Excessive API Calls
- **Databento**: $0.01 per call × 100,000 = $1,000/month
- **ScrapeCreators**: Unlimited plans can be expensive
- **Risk**: **$500-1,000/month** if not rate-limited

### 4. Storage Growth Over Time
- **5 years**: ~115 GB
- **10 years**: ~230 GB
- **20 years**: ~460 GB
- **Cost**: Grows linearly (~$9/month at 20 years)

---

## 🛡️ Cost Protection Strategies

### 1. Query Optimization (Critical!)
```sql
-- ✅ GOOD: Partitioned + Date Filter
SELECT * FROM `cbi-v15.features.daily_ml_matrix`
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  AND symbol = 'ZL'

-- ❌ BAD: Full Table Scan
SELECT * FROM `cbi-v15.features.daily_ml_matrix`
WHERE symbol = 'ZL'  -- No date filter!
```

**Savings**: 80-90% reduction in query costs

### 2. Rate Limiting
- **API calls**: Implement rate limiting
- **Scraping**: Respect robots.txt, add delays
- **BigQuery**: Use query slots, limit concurrency

**Savings**: Prevent runaway costs

### 3. Budget Alerts
```bash
# Set multiple thresholds
gcloud billing budgets create \
    --budget-amount=50USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=80 \
    --threshold-rule=percent=100 \
    --threshold-rule=percent=120
```

**Protection**: Early warning system

### 4. Cost Monitoring Dashboard
- **Daily cost reports**: Email alerts
- **BigQuery usage**: Track by dataset
- **API usage**: Track by source

**Protection**: Catch issues early

---

## 📈 Cost Scaling by Data Volume

| Data Volume | Monthly Cost | Notes |
|-------------|-------------|-------|
| **Low** (current) | ~$7 | Normal usage |
| **Medium** (2x) | ~$15 | 2x data/queries |
| **High** (5x) | ~$50 | Heavy usage |
| **Very High** (10x) | ~$130 | Extreme usage |
| **Extreme** (20x) | ~$850 | Worst case |

---

## 🎯 Recommended Budget Thresholds

### Conservative Budget
- **Monthly Budget**: $25/month
- **Alert at**: $20/month (80%)
- **Good for**: Normal to medium usage

### Realistic Budget
- **Monthly Budget**: $50/month
- **Alert at**: $40/month (80%)
- **Good for**: Medium to high usage

### Extreme High Budget
- **Monthly Budget**: $250/month
- **Alert at**: $200/month (80%)
- **Good for**: Maximum usage + external APIs

---

## ✅ Cost Summary

### Normal Usage
- **GCP Costs**: ~$7/month
- **External APIs**: $0-200/month (varies by plan)
- **Total**: **~$7-207/month**

### Extreme High Usage
- **GCP Costs**: ~$128/month
- **External APIs**: ~$720/month
- **Total**: **~$848/month**

### Most Likely High Scenario
- **GCP Costs**: ~$26/month
- **External APIs**: ~$200/month
- **Total**: **~$226/month**

---

## 🚨 Critical Cost Controls

1. **Always partition by date** (reduces query costs 80-90%)
2. **Always use date filters** (prevents full table scans)
3. **Always use us-central1** (avoids 2-3x multi-region costs)
4. **Set budget alerts** ($25, $50, $250 thresholds)
5. **Monitor BigQuery usage daily** (biggest cost driver)
6. **Rate limit API calls** (prevents runaway costs)
7. **Archive old data** (use long-term storage discount)

---

**Last Updated**: November 28, 2025

