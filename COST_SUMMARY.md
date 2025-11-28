# CBI-V15 Cost Summary

**Quick Reference**: Monthly GCP costs for CBI-V15

---

## 💰 Monthly Cost: ~$6.00/month

### Breakdown

| Service | Cost | Notes |
|---------|------|-------|
| **BigQuery** | $5.55 | Storage ($2.30) + Queries ($3.25) |
| **Secret Manager** | $0.24 | 4 secrets × $0.06 |
| **Cloud Storage** | $0.20 | Model artifacts, exports |
| **Everything Else** | $0.00 | All within free tiers ✅ |
| **TOTAL** | **$6.00** | |

---

## 📊 Annual Cost: ~$72/year

---

## 💡 Cost Optimization

- ✅ All tables partitioned by date
- ✅ All tables clustered by symbol
- ✅ Stay within BigQuery free tier (1 TB queries/month)
- ✅ Use long-term storage discount (90+ days)
- ✅ **CRITICAL**: Always use `us-central1` only!

---

## ⚠️ Cost Warnings

1. **Multi-region costs 2-3x more** - Always use `us-central1`
2. **Unpartitioned queries expensive** - Always partition by date
3. **Full table scans costly** - Always use date filters

---

## 🎯 Comparison

- **V14**: ~$250/month ❌
- **V15**: ~$6/month ✅
- **Savings**: 97% reduction! 🎉

---

**Full Analysis**: See [docs/setup/GCP_COST_ANALYSIS.md](docs/setup/GCP_COST_ANALYSIS.md)

