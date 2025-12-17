# CBI-V15.1 Implementation Plan - Complete Index

**Created:** 2025-01-02  
**Total Tasks:** 49 across 6 phases  
**Estimated Time:** 59-81 hours (2-3 weeks)

---

## 📚 PHASE DOCUMENTS

### Individual Phase Files (Detailed)

1. **[PHASE_0_DETAILED.md](./PHASE_0_DETAILED.md)** — Critical Infrastructure & Bug Fixes (13 tasks, 8-12 hours)
2. **[PHASE_1_DETAILED.md](./PHASE_1_DETAILED.md)** — Critical Data Feeds (7 tasks, 12-16 hours)
3. **[PHASE_2_TO_5_DETAILED_PLAN.md](./PHASE_2_TO_5_DETAILED_PLAN.md)** — AutoGluon Integration (9 tasks, 8-10 hours)
4. **[PHASE_3_TO_5_CONTINUED.md](./PHASE_3_TO_5_CONTINUED.md)** — Bucket Specialists + Ensemble + Orchestration (20 tasks, 30-44 hours)

### Summary Documents

- **[IMPLEMENTATION_PLAN_SUMMARY.md](./IMPLEMENTATION_PLAN_SUMMARY.md)** — Executive summary, timeline, critical success factors
- **[ALL_PHASES_INDEX.md](./ALL_PHASES_INDEX.md)** — This file (complete index)

---

## 🎯 QUICK REFERENCE

### Phase 0: Critical Infrastructure (13 tasks)

**Status:** IN PROGRESS  
**File:** [PHASE_0_DETAILED.md](./PHASE_0_DETAILED.md)

**Key Tasks:**

- Fix Databento column name (`date` → `as_of_date`)
- Fix FRED table reference (`fred_observations` → `fred_economic`)
- Fix EIA table reference
- Create local DuckDB mirror for Mac M4
- Create MotherDuck → Local sync script
- Remove all TSci dependencies
- Add ON CONFLICT idempotency logic

**Validation:** `bash scripts/system_status.sh` shows all green

---

### Phase 1: Critical Data Feeds (7 tasks)

**Status:** NOT STARTED  
**File:** [PHASE_1_DETAILED.md](./PHASE_1_DETAILED.md)

**Key Tasks:**

- EPA RIN Prices Trigger job (FREE, weekly)
- Remove mock USDA data
- CFTC COT Trigger job
- FRED Daily Ingest
- Farm Policy News scraper (MANDATORY)
- farmdoc Daily scraper

**Big 8 Coverage:** All 8 buckets validated

---

### Phase 2: AutoGluon Integration (9 tasks)

**Status:** NOT STARTED  
**File:** [PHASE_2_TO_5_DETAILED_PLAN.md](./PHASE_2_TO_5_DETAILED_PLAN.md)

**Key Tasks:**

- Install AutoGluon 1.4 on Mac M4 (libomp fix)
- Create TabularPredictor wrapper (quantile regression)
- Create TimeSeriesPredictor wrapper (Chronos-Bolt)
- Configure foundation models (Mitra, TabPFNv2, TabICL, TabM)
- Create feature drift detection

**Foundation Models:** Mitra, TabPFNv2, TabICL, TabM, Chronos-Bolt (all CPU-compatible)

---

### Phase 3: Bucket Specialist Infrastructure (5 tasks)

**Status:** NOT STARTED  
**File:** [PHASE_3_TO_5_CONTINUED.md](./PHASE_3_TO_5_CONTINUED.md) (lines 1-150)

**Key Tasks:**

- Create `bucket_feature_selectors.yaml` config
- Create 8 bucket training configs
- Create `bucket_specialist.py` trainer
- Create `train_all_buckets.py` orchestrator

**Output:** 8 bucket specialists + 1 main predictor + 1 Chronos baseline = 10 L0 models

---

### Phase 4: AutoGluon Stacking & Monte Carlo (7 tasks)

**Status:** NOT STARTED  
**File:** [PHASE_3_TO_5_CONTINUED.md](./PHASE_3_TO_5_CONTINUED.md) (lines 253-619)

**Key Tasks:**

- Create L1 stacking layer
- Create L2.5 greedy weighted ensemble (UPGRADED FEATURE)
- Create L3 Monte Carlo simulation (VaR/CVaR)
- Create full pipeline orchestrator

**Architecture:**

```
L0: 10 models → training.bucket_predictions
L1: AutoGluon stacking layer
L2: WeightedEnsemble_L2 (automatic)
L2.5: Greedy weighted ensemble
L3: Monte Carlo (10K scenarios)
Final: forecasts.zl_predictions
```

---

### Phase 5: Trigger.dev Orchestration (7 tasks)

**Status:** NOT STARTED  
**File:** [PHASE_3_TO_5_CONTINUED.md](./PHASE_3_TO_5_CONTINUED.md) (lines 621-1060)

**Key Tasks:**

- Daily training job (2 AM UTC)
- Daily forecast job (10 AM UTC)
- Model monitoring job (6 PM UTC)
- Weekly retraining job (Sunday midnight)
- Data quality monitoring job (hourly)
- Notification system (Slack/email)

**Trigger.dev Jobs:** 5 total (training, forecast, monitoring, retraining, data quality)

---

## 🗂️ DATABASE ARCHITECTURE UPDATES

### New Tables Created

1. **`database/models/04_training/bucket_predictions.sql`**
   - L0 bucket specialist OOF predictions
   - Used by L1 stacking layer
   - Columns: as_of_date, bucket_name, horizon, q10, q50, q90, actual

2. **`database/models/07_forecasts/zl_predictions.sql`**
   - Final ensemble predictions + VaR/CVaR
   - Used by dashboard and monitoring
   - Columns: as_of_date, forecast_date, q10, q50, q90, var_95, var_99, cvar_95, cvar_99, Big 8 scores

### Updated Files

- **`database/models/MANIFEST.md`** — Updated to 25 SQL files (was 23)
- Added 07_forecasts/ folder
- Added AutoGluon architecture documentation

---

## ⏱️ TIMELINE

| Phase     | Tasks  | Hours     | Risk     | Status          |
| --------- | ------ | --------- | -------- | --------------- |
| Phase -1  | 6      | 1         | LOW      | IN PROGRESS     |
| Phase 0   | 13     | 8-12      | HIGH     | IN PROGRESS     |
| Phase 1   | 7      | 12-16     | MEDIUM   | NOT STARTED     |
| Phase 2   | 9      | 8-10      | HIGH     | NOT STARTED     |
| Phase 3   | 5      | 10-14     | MEDIUM   | NOT STARTED     |
| Phase 4   | 7      | 12-16     | HIGH     | NOT STARTED     |
| Phase 5   | 7      | 8-12      | MEDIUM   | NOT STARTED     |
| **TOTAL** | **49** | **59-81** | **HIGH** | **IN PROGRESS** |

**Calendar Time:** 2-3 weeks (assuming 4-6 hours/day)

---

## 🚨 CRITICAL PATH

```
Phase 0 (Infrastructure) → Phase 1 (Data) → Phase 2 (AutoGluon) →
Phase 3 (Buckets) → Phase 4 (Ensemble) → Phase 5 (Orchestration)
```

**⚠️ HARD STOP:** Each phase MUST complete and validate before proceeding

---

## 📖 HOW TO USE THIS PLAN

1. **Start with Summary:** Read [IMPLEMENTATION_PLAN_SUMMARY.md](./IMPLEMENTATION_PLAN_SUMMARY.md)
2. **Execute Phase 0:** Follow [PHASE_0_DETAILED.md](./PHASE_0_DETAILED.md) step-by-step
3. **Validate Phase 0:** Run `bash scripts/system_status.sh`
4. **Execute Phase 1:** Follow [PHASE_1_DETAILED.md](./PHASE_1_DETAILED.md)
5. **Continue Phases 2-5:** Follow combined files for remaining phases
6. **Final Validation:** Use checklist in Phase 5 document

---

**🎯 GOAL:** Production-ready AutoGluon hybrid forecasting system for ZL (soybean oil) futures
