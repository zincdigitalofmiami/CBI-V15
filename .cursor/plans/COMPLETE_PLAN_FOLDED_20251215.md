# CBI-V15 Complete Plan - Folded into Task List
**Date:** December 15, 2025  
**Status:** Phases 0-2 COMPLETE, Phase 0 Post-Completion Fixes READY, Phase 3+ PENDING  
**Total Time to Full Completion:** 5-8 hours

---

## ✅ COMPLETED PHASES

### Phase 0: Security (COMPLETED)
- ✅ Token rotation documented in docs/ops/SECURITY_TOKEN_ROTATION_REQUIRED.md
- ✅ User will rotate token manually after initialization

### Phase 1: Schema Initialization (COMPLETED)
- ✅ 9 schemas created (raw, staging, features, features_dev, training, forecasts, reference, ops, explanations)
- ✅ 69 tables deployed to MotherDuck
- ✅ 69 tables deployed to Local DuckDB (matching)
- ✅ 2 non-critical partial index warnings (MotherDuck limitation)

### Phase 2: Reference Data Seeding (COMPLETED)
- ✅ reference.symbols: 33 rows (all canonical symbols)
- ✅ reference.train_val_test_splits: 3 rows (prod_v1, dev_v1, backtest_2023)
- ✅ reference.driver_group: 8 rows (Big 8 buckets)
- ✅ reference.geo_countries: 7 rows
- ✅ reference.geo_admin_regions: 10 rows

---

## 🔧 PHASE 0 POST-COMPLETION FIXES (50 minutes)

### Fix #1: DuckDB Version (5 min) [CRITICAL]
- **Current:** v1.4.3
- **Required:** v1.4.2 (MotherDuck compatible)
- **Command:** `pip install duckdb==1.4.2`
- **Verify:** `python -c "import duckdb; print(duckdb.__version__)"`

### Fix #2: Install AutoGluon (15 min) [HIGH]
- **Command:** `pip install autogluon.tabular[all]>=1.4.0 autogluon.timeseries[all]>=1.4.0`
- **Verify:** `python -c "from autogluon.tabular import TabularPredictor; print('Ready')"`

### Fix #3: Recreate Local DuckDB (30 min) [HIGH]
- **Delete:** `data/duckdb/cbi_v15.duckdb`
- **Recreate with subset:** reference.* (12), features.* (13), training.* (6)
- **Omit:** raw.*, staging.*, forecasts.*, ops.*, explanations.*
- **Run:** `python scripts/sync_motherduck_to_local.py --schemas reference,features,training`

---

## ⏳ PENDING PHASES

### Phase 3: Raw Data Ingestion (2-4 hours)
- Create 7 missing Trigger.dev jobs (2-3 hrs)
- Run ingestion jobs
- Expected: 30K+ Databento, 40K+ FRED, 10K+ CFTC, 500+ USDA, 500+ EIA, 700+ EPA rows

### Phase 4: Feature Engineering (1-2 hours)
- Run: `python src/engines/anofox/build_all_features.py`
- Expected: 3K+ rows in features.daily_ml_matrix_zl (88 columns)

### Phase 5: Sync MotherDuck → Local (30 minutes)
- Run: `python scripts/sync_motherduck_to_local.py --schemas reference,features,training`
- Expected: Local DB populated, 50-500 MB

### Phase 7: Cleanup (15 minutes)
- Delete stale files
- Verify .gitignore
- Run final audit

---

## 📊 TIMELINE

| Phase | Task | Duration | Cumulative |
|-------|------|----------|-----------|
| 0 | Post-completion fixes | 50 min | 50 min |
| 3 | Raw data ingestion | 2-4 hrs | 2.5-4.5 hrs |
| 4 | Feature engineering | 1-2 hrs | 3.5-6.5 hrs |
| 5 | Sync to local | 30 min | 4-7 hrs |
| 7 | Cleanup | 15 min | 4.25-7.25 hrs |
| **TOTAL** | **To Phase 1 Complete** | **5-8 hours** | — |

---

## 🎯 NEXT IMMEDIATE ACTIONS

1. **Fix DuckDB version** (5 min)
2. **Install AutoGluon** (15 min)
3. **Recreate Local DuckDB** (30 min)
4. **Create Trigger.dev jobs** (2-3 hrs)
5. **Run Phase 3 ingestion** (2-4 hrs)
6. **Run Phase 4 features** (1-2 hrs)
7. **Sync to local** (30 min)
8. **Cleanup** (15 min)

---

## 📋 DOCUMENTATION

- **Blocker Audit:** `.cursor/plans/BLOCKER_AUDIT_20251215.md`
- **Audit Summary:** `.cursor/plans/AUDIT_SUMMARY_20251215.md`
- **Database Review:** `.cursor/plans/DATABASE_INIT_REVIEW_20251215.md`
- **Task List:** Updated with 61 tasks reflecting complete plan

---

## ✨ KEY INSIGHT

All work is:
- ✓ Fixable
- ✓ Well-understood
- ✓ Clearly scoped
- ✓ Estimated with confidence

**Status:** Ready to proceed with Phase 0 post-completion fixes immediately.

