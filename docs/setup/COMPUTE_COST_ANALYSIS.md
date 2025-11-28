# Compute Cost Analysis - Data Ingestion & Dashboard

**Date**: November 28, 2025  
**Scope**: ZL-only, Vegas Intel, Dashboard compute  
**Budget**: $50/month (GCP only)

---

## 💻 Compute Cost Breakdown

### 1. Databento Price Data Ingestion (1-Hour Pulls)

#### Compute Requirements
- **Frequency**: 24 pulls/day (ZL price)
- **Other Symbols**: 6 pulls/day (ZS, ZM, CL, HO, FCPO every 4 hours)
- **Total Pulls**: 30 pulls/day = 900/month
- **Compute Time**: ~5 seconds per pull
- **Memory**: 512 MB per invocation

#### Cloud Functions Cost
- **Invocations**: 900/month
- **Free Tier**: First 2M/month **FREE** ✅
- **Compute Time**: 900 × 5 seconds × 0.5 GB = 2,250 GB-seconds/month
- **Free Tier**: First 400,000 GB-seconds/month **FREE** ✅
- **Cost**: **$0.00/month** (within free tier)

#### Cloud Run Alternative (If Needed)
- **CPU**: 1 vCPU
- **Memory**: 512 MB
- **Duration**: 5 seconds per pull
- **Cost**: ~$0.000024 per invocation
- **Monthly**: 900 × $0.000024 = **$0.02/month**

**Databento Ingestion**: **$0.00-0.02/month** ✅

---

### 2. Glide Data Ingestion (Daily)

#### Current Schedule
- **Frequency**: Every 6 hours (4 pulls/day)
- **Updated**: **Daily** (1 pull/day) ✅

#### Compute Requirements
- **Frequency**: 1 pull/day = 30/month
- **Compute Time**: ~10 seconds per pull
- **Memory**: 512 MB per invocation

#### Cloud Functions Cost
- **Invocations**: 30/month
- **Free Tier**: First 2M/month **FREE** ✅
- **Compute Time**: 30 × 10 seconds × 0.5 GB = 150 GB-seconds/month
- **Free Tier**: First 400,000 GB-seconds/month **FREE** ✅
- **Cost**: **$0.00/month** (within free tier)

**Glide Ingestion**: **$0.00/month** ✅

---

### 3. Vegas Events Ingestion (NEW)

#### Compute Requirements
- **Frequency**: Daily (1 pull/day) = 30/month
- **Compute Time**: ~15 seconds per pull (more data)
- **Memory**: 1 GB per invocation

#### Cloud Functions Cost
- **Invocations**: 30/month
- **Free Tier**: First 2M/month **FREE** ✅
- **Compute Time**: 30 × 15 seconds × 1 GB = 450 GB-seconds/month
- **Free Tier**: First 400,000 GB-seconds/month **FREE** ✅
- **Cost**: **$0.00/month** (within free tier)

**Vegas Events Ingestion**: **$0.00/month** ✅

---

### 4. Dashboard API Compute (Cloud Run)

#### ZL Forecast Page
- **Requests**: ~1,000/day = 30,000/month
- **CPU**: 0.5 vCPU
- **Memory**: 512 MB
- **Duration**: 200ms per request
- **Cost**: ~$0.000001 per request
- **Monthly**: 30,000 × $0.000001 = **$0.03/month**

#### Vegas Intel Page
- **Requests**: ~500/day = 15,000/month
- **CPU**: 0.5 vCPU
- **Memory**: 512 MB
- **Duration**: 300ms per request (more data)
- **Cost**: ~$0.0000015 per request
- **Monthly**: 15,000 × $0.0000015 = **$0.02/month**

#### Other Pages (Signals, Regimes, etc.)
- **Requests**: ~500/day = 15,000/month
- **CPU**: 0.5 vCPU
- **Memory**: 512 MB
- **Duration**: 200ms per request
- **Cost**: ~$0.000001 per request
- **Monthly**: 15,000 × $0.000001 = **$0.02/month**

**Dashboard API Total**: **$0.07/month**

---

### 5. Data Processing Compute (Cloud Functions)

#### Feature Engineering (Daily)
- **Frequency**: Daily (after data ingestion)
- **Compute Time**: ~30 seconds
- **Memory**: 2 GB
- **Invocations**: 30/month
- **Compute Time**: 30 × 30 seconds × 2 GB = 1,800 GB-seconds/month
- **Free Tier**: First 400,000 GB-seconds/month **FREE** ✅
- **Cost**: **$0.00/month** (within free tier)

**Data Processing**: **$0.00/month** ✅

---

### 6. Model Inference Compute (Cloud Run - Optional)

#### If Running Models on Cloud Run (Not Recommended - Mac Preferred)
- **Requests**: ~100/day = 3,000/month
- **CPU**: 2 vCPU
- **Memory**: 4 GB
- **Duration**: 2 seconds per request
- **Cost**: ~$0.0001 per request
- **Monthly**: 3,000 × $0.0001 = **$0.30/month**

**Note**: Models run on Mac M4, not Cloud Run (per architecture)

**Model Inference**: **$0.00/month** (Mac-based) ✅

---

## 📊 Total Compute Costs

### Ingestion Compute
| Service | Cost |
|---------|------|
| Databento Ingestion | $0.00/month |
| Glide Ingestion | $0.00/month |
| Vegas Events Ingestion | $0.00/month |
| Data Processing | $0.00/month |
| **Subtotal** | **$0.00/month** |

### Dashboard Compute
| Service | Cost |
|---------|------|
| ZL Forecast Page | $0.03/month |
| Vegas Intel Page | $0.02/month |
| Other Pages | $0.02/month |
| **Subtotal** | **$0.07/month** |

### Total Compute Costs
- **Ingestion**: $0.00/month (all within free tier)
- **Dashboard**: $0.07/month
- **TOTAL**: **$0.07/month** ✅

---

## 🔧 Updated Configuration

### Glide Schedule (Updated to Daily)
```yaml
glide:
  api_key_env: GLIDE_API_KEY
  rate_limit: 100
  schedule: "0 2 * * *"  # Daily at 2 AM UTC
  endpoints: [restaurants, casinos, shifts, events]
```

### Vegas Events (NEW)
```yaml
vegas_events:
  api_key_env: GLIDE_API_KEY
  rate_limit: 100
  schedule: "0 3 * * *"  # Daily at 3 AM UTC (after Glide)
  endpoints: [events]  # Vegas events only
```

---

## 💰 Revised Total Cost Breakdown

### GCP Costs (Including Compute)

| Service | Cost |
|---------|------|
| **BigQuery Storage** | $0.50/month |
| **BigQuery Queries** | $0.75/month |
| **Public Dataset Queries** | $0.25/month |
| **Cloud Storage** | $0.10/month |
| **Secret Manager** | $0.24/month |
| **Compute (Ingestion)** | $0.00/month |
| **Compute (Dashboard)** | $0.07/month |
| **TOTAL** | **$1.91/month** ✅ |

### Budget Compliance
- **Budget Cap**: $50/month
- **Actual Cost**: $1.91/month
- **Compliance**: ✅ **96% under budget!**

---

## 🎯 Cost Scaling (Compute)

### Normal Usage
- **Compute**: $0.07/month
- **Total GCP**: $1.91/month

### High Usage (10x traffic)
- **Dashboard Requests**: 10x = 600,000/month
- **Compute**: ~$0.70/month
- **Total GCP**: ~$2.54/month

### Very High Usage (100x traffic)
- **Dashboard Requests**: 100x = 6M/month
- **Compute**: ~$7.00/month
- **Total GCP**: ~$8.84/month

**Still well under $50/month budget!** ✅

---

## 🚨 Compute Optimization Strategies

### 1. Cloud Functions (Ingestion)
- ✅ Use Cloud Functions (free tier)
- ✅ Keep invocations <2M/month
- ✅ Keep compute <400K GB-seconds/month
- ✅ Current: Well within limits ✅

### 2. Cloud Run (Dashboard)
- ✅ Use Cloud Run (pay-per-use)
- ✅ Enable request concurrency (handle multiple requests)
- ✅ Use Cloud CDN for static assets (reduces compute)
- ✅ Cache API responses (reduces compute)

### 3. Caching Strategy
- ✅ **API Response Caching**: 5-minute cache for dashboard queries
- ✅ **CDN Caching**: Static assets cached at edge
- ✅ **Materialized Views**: Pre-compute dashboard data
- **Savings**: 80-90% reduction in compute costs

---

## 📋 Updated Files Needed

### 1. Update `config/ingestion/sources.yaml`
- Glide: Daily (not every 6 hours)
- Add Vegas Events: Daily

### 2. Create `config/dashboard/pages.yaml`
- ZL Forecast page config
- Vegas Intel page config
- Other pages config

### 3. Create `scripts/ingestion/vegas/collect_vegas_events.py`
- Vegas Events ingestion script

---

## ✅ Summary

### Compute Costs
- **Ingestion**: $0.00/month (all within free tier)
- **Dashboard**: $0.07/month
- **Total Compute**: **$0.07/month**

### Total GCP Costs (Including Compute)
- **Storage/Queries**: $1.84/month
- **Compute**: $0.07/month
- **TOTAL**: **$1.91/month** ✅

### Budget Compliance
- **Budget Cap**: $50/month
- **Actual Cost**: $1.91/month
- **Compliance**: ✅ **96% under budget!**

---

**Last Updated**: November 28, 2025

