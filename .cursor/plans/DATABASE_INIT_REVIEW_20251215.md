# CBI-V15 Database Initialization Review
**Date:** December 15, 2025  
**Status:** 3 of 7 phases complete, 7 critical blockers identified

---

## ✅ COMPLETED (3 Phases)

| Phase | Status | Details |
|-------|--------|---------|
| **Phase 0: Security** | ✅ | Token rotation documented |
| **Phase 1: Schema Init** | ✅ | 9 schemas, 69 tables (MotherDuck + Local) |
| **Phase 2: Reference Data** | ✅ | 33 symbols, 3 splits, 8 buckets, geo data |

---

## ⏳ PENDING (4 Phases)

| Phase | Blocker | Impact |
|-------|---------|--------|
| **Phase 3: Raw Ingestion** | DuckDB v1.4.3 ↔ MotherDuck v1.4.2 mismatch | Can't verify data loaded |
| **Phase 4: Features** | Depends on Phase 3 | Can't build features |
| **Phase 5: Sync Local** | Missing `sync_motherduck_to_local.py` | Can't train locally |
| **Phase 7: Cleanup** | Depends on all phases | Final validation pending |

---

## 🔴 CRITICAL BLOCKERS (7)

1. **DuckDB Version Mismatch** → `pip install duckdb==1.4.2`
2. **Sync Script Missing** → Create `scripts/sync_motherduck_to_local.py`
3. **No Raw Data** → Phase 3 ingestion jobs not run yet
4. **Local DB Over-Provisioned** → Has 69 tables, should have ~30 (reference+features+training only)
5. **AutoGluon Not Installed** → Phase 2 tasks not started
6. **Feature Engineering Blocked** → Depends on raw data
7. **Trigger.dev Jobs** → Some not fully integrated

---

## 🎯 IMMEDIATE ACTIONS (Next 24 Hours)

### CRITICAL (Unblock everything)
```bash
# 1. Fix DuckDB version
pip install duckdb==1.4.2

# 2. Create sync script
# File: scripts/sync_motherduck_to_local.py
# Purpose: Copy reference.*, features.*, training.* from MotherDuck → Local
```

### HIGH (Enable data flow)
```bash
# 3. Run Phase 3 ingestion jobs
npx trigger.dev@latest dev
# Trigger: databento_ingest_job, fred_daily_ingest, cftc_cot_ingest

# 4. Run Phase 4 feature engineering
python src/engines/anofox/build_all_features.py

# 5. Run sync script
python scripts/sync_motherduck_to_local.py
```

---

## 📊 TASK LIST STATUS

- ✅ **8 tasks completed** (FORENSIC, FRED fixes, schema validation)
- [/] **1 task in progress** (EIA/EPA split)
- [ ] **~100 tasks not started** (Phases 0, 1, 2, 3, 4, 5, -1)

---

## ⏱️ TIME TO FULL SYSTEM

- Phase 3 (Ingestion): 2-4 hours
- Phase 4 (Features): 1-2 hours
- Phase 5 (Sync): 30 minutes
- Phase 2 (AutoGluon): 2-3 hours
- **Total: 6-12 hours**

---

## 🏗️ LOCAL DB ARCHITECTURE CORRECTION

**Current:** 69 tables (mirrors MotherDuck exactly)  
**Correct:** ~30 tables (reference + features + training only)

**Should OMIT locally:**
- raw.* (ingestion-only)
- staging.* (intermediate)
- forecasts.* (dashboard-only)
- ops.* (monitoring-only)
- explanations.* (SHAP-only)

**Should KEEP locally:**
- reference.* (symbols, splits, regimes)
- features.* (daily_ml_matrix, bucket_*)
- training.* (bucket_predictions, meta_ml_matrix)

