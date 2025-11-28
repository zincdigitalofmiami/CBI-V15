# Audit Fixes Required - Before BigQuery Setup

**Date**: November 28, 2025  
**Status**: ⚠️ **CRITICAL** - Must fix before proceeding

---

## 🚨 Critical Gaps Identified

### 1. Missing Tables (10 critical tables) ⚠️

**News/Sentiment Buckets** (4 tables):
- ⚠️ `raw.scrapecreators_news_buckets`
- ⚠️ `staging.news_bucketed`
- ⚠️ `staging.sentiment_buckets`
- ⚠️ `features.sentiment_features_daily`

**Regime System** (3 tables):
- ⚠️ `reference.regime_calendar`
- ⚠️ `reference.regime_weights`
- ⚠️ `features.regime_indicators_daily`

**Neural Features** (3 tables):
- ⚠️ `features.neural_signals_daily`
- ⚠️ `features.neural_master_score`
- ⚠️ `reference.neural_drivers`

**Action**: ✅ **ADD** - Created `dataform/definitions/00_skeleton/missing_critical_tables.sqlx`

---

### 2. Missing Scheduler Configuration ⚠️

**Status**: ⚠️ **NOT PLANNED**

**Action**: ✅ **ADD** - Created `config/schedulers/ingestion_schedules.yaml`

---

### 3. Missing Segmentation Strategy ⚠️

**Status**: ⚠️ **NOT DOCUMENTED**

**Action**: ✅ **ADD** - Created `docs/architecture/NEWS_NEURAL_SEGMENTATION_STRATEGY.md`

---

### 4. Missing Workflow Documentation ⚠️

**Status**: ⚠️ **NOT DOCUMENTED**

**Action**: ✅ **ADD** - Created `docs/architecture/SCHEDULER_WORKFLOW.md`

---

## ✅ Fixes Applied

### Files Created:

1. ✅ `dataform/definitions/00_skeleton/missing_critical_tables.sqlx` - 10 missing tables
2. ✅ `config/schedulers/ingestion_schedules.yaml` - Scheduler configuration
3. ✅ `docs/architecture/NEWS_NEURAL_SEGMENTATION_STRATEGY.md` - Segmentation strategy
4. ✅ `docs/architecture/SCHEDULER_WORKFLOW.md` - Workflow documentation

---

## 📋 Remaining Actions

### Before BigQuery Setup:

1. ⚠️ **REVIEW** missing tables skeleton (verify structure)
2. ⚠️ **IMPLEMENT** segmentation logic in ingestion scripts
3. ⚠️ **CREATE** completion tracking table (`ops.ingestion_completion`)
4. ⚠️ **UPDATE** skeleton structure to include all 10 missing tables

### After BigQuery Setup:

1. ⚠️ **TEST** segmentation at ingestion
2. ⚠️ **VERIFY** scheduler workflows
3. ⚠️ **MONITOR** drift detection
4. ⚠️ **VALIDATE** regime weighting

---

## ✅ Verification Checklist

### Tables:
- [ ] ✅ All 10 missing tables added to skeleton
- [ ] ✅ Partitioning/clustering verified
- [ ] ✅ No joins in skeleton structure

### Schedulers:
- [ ] ✅ All 9 schedulers configured
- [ ] ✅ Workflow documented
- [ ] ✅ Coordination strategy defined

### Segmentation:
- [ ] ✅ Bucket segmentation at ingestion
- [ ] ✅ Temporal segmentation (regime tagging)
- [ ] ✅ Source segmentation (trust scoring)
- [ ] ✅ Volume normalization
- [ ] ✅ Neural layer segmentation

---

**Status**: ⚠️ **AUDIT COMPLETE** - 10 missing tables identified, fixes applied

**Recommendation**: ⚠️ **REVIEW FIXES** before proceeding with BigQuery setup

---

**Last Updated**: November 28, 2025

