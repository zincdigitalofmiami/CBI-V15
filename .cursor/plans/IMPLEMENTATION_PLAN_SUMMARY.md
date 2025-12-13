# CBI-V15.1 Implementation Plan Summary

**Created:** 2025-01-02  
**Status:** Ready for execution  
**Architecture:** AutoGluon 1.4 Hybrid (Big 8 Tabular + Core ZL TimeSeries + Meta + Greedy Ensemble + Monte Carlo)

---

## 📚 PLAN DOCUMENTS (READ IN ORDER)

This implementation plan is split across 3 documents for clarity:

1. **`.cursor/plans/PHASE_0_TO_5_DETAILED_PLAN.md`** — Phase 0 (Critical Infrastructure) + Phase 1 (Data Feeds)
2. **`.cursor/plans/PHASE_2_TO_5_DETAILED_PLAN.md`** — Phase 2 (AutoGluon Integration)
3. **`.cursor/plans/PHASE_3_TO_5_CONTINUED.md`** — Phase 3 (Bucket Specialists) + Phase 4 (Ensemble) + Phase 5 (Orchestration)

---

## 🎯 EXECUTIVE SUMMARY

### Total Scope
- **49 tasks** across **6 phases** (Phase -1 through Phase 5)
- **10 ML models** (8 bucket specialists + 1 main predictor + 1 Chronos baseline)
- **5-layer ensemble** (L0 → L1 → L2 → L2.5 greedy → L3 Monte Carlo)
- **8 data sources** (Databento, FRED, EIA, EPA, USDA, CFTC, Farm Policy News, farmdoc Daily)
- **5 Trigger.dev jobs** (daily training, daily forecast, monitoring, weekly retraining, data quality)

### Critical Path
```
Phase 0 (Infrastructure) → Phase 1 (Data) → Phase 2 (AutoGluon) → 
Phase 3 (Buckets) → Phase 4 (Ensemble) → Phase 5 (Orchestration)
```

**⚠️ HARD STOP:** Each phase MUST complete before proceeding to next phase.

---

## 📊 PHASE BREAKDOWN

### Phase -1: Repo Scaffolding (6 tasks)
**Status:** IN PROGRESS  
**Goal:** Create missing directories and scaffolding  
**Key Tasks:**
- Create `docs/_archive/` directory
- Create missing `src/` directories
- Create `pyproject.toml`
- Create training scaffolding scripts

**Estimated Time:** 1 hour

---

### Phase 0: Critical Infrastructure & Bug Fixes (13 tasks)
**Status:** NOT STARTED  
**Goal:** Fix critical data pipeline bugs and establish local DuckDB mirror  
**Key Tasks:**
- Fix Databento column name (`date` → `as_of_date`)
- Fix FRED table reference (`fred_observations` → `fred_economic`)
- Fix EIA table reference (`eia_biofuels` → `eia_petroleum`)
- Add missing FRED series (DFEDTARU, VIXCLS)
- Create local DuckDB mirror for Mac M4 training
- Create MotherDuck → Local sync script
- Add ON CONFLICT idempotency logic
- Remove all TSci dependencies
- Parameterize environment variables

**Estimated Time:** 8-12 hours  
**Risk Level:** HIGH (blocks all downstream work)

**Validation:** `bash scripts/system_status.sh` shows all green checks

---

### Phase 1: Critical Data Feeds (7 tasks)
**Status:** NOT STARTED  
**Goal:** Implement missing data sources for Big 8 bucket coverage  
**Key Tasks:**
- Create EPA RIN Prices Trigger job (FREE, weekly data)
- Remove mock data from USDA Export Sales
- Create CFTC COT Trigger job
- Create FRED Daily Ingest Trigger job
- Create Farm Policy News scraper (MANDATORY for China/Tariff buckets)
- Create farmdoc Daily scraper (Scott Irwin RIN models)

**Estimated Time:** 12-16 hours  
**Risk Level:** MEDIUM (data availability dependent)

**Big 8 Coverage:**
- ✅ Crush: Databento (ZL/ZS/ZM), NOPA, farmdoc Grain Outlook
- ✅ China: Farm Policy News Trade, USDA Export Sales, farmdoc Trade
- ✅ FX: FRED FX series, Databento (6L/DX)
- ✅ Fed: FRED rates/curve, Farm Policy News Budget
- ✅ Tariff: Farm Policy News Trade, ScrapeCreators Trump
- ✅ Biofuel: EPA RIN Prices, EIA, farmdoc RINs
- ✅ Energy: EIA petroleum, Databento (CL/HO/RB)
- ✅ Volatility: FRED VIXCLS, Databento VIX, STLFSI4

---

### Phase 2: AutoGluon Integration (9 tasks)
**Status:** NOT STARTED  
**Goal:** Integrate AutoGluon 1.4 with Mac M4 compatibility  
**Key Tasks:**
- Install AutoGluon 1.4 on Mac M4 (libomp fix)
- Create TabularPredictor wrapper (quantile regression)
- Create TimeSeriesPredictor wrapper (Chronos-Bolt)
- Configure foundation models (Mitra, TabPFNv2, TabICL, TabM)
- Create feature drift detection module
- Document AutoGluon foundation models

**Estimated Time:** 8-10 hours  
**Risk Level:** HIGH (Mac M4 compatibility issues)

**Foundation Models (CPU-compatible):**
- Mitra (tabular foundation model)
- TabPFNv2 (prior-fitted network)
- TabICL (in-context learning)
- TabM (tabular transformer)
- Chronos-Bolt (zero-shot time series)

---

### Phase 3: Bucket Specialist Infrastructure (5 tasks)
**Status:** NOT STARTED  
**Goal:** Build 8 bucket specialist trainers with feature selection  
**Key Tasks:**
- Create `bucket_feature_selectors.yaml` config
- Create 8 bucket training configs (crush, china, fx, fed, tariff, biofuel, energy, volatility)
- Create `bucket_specialist.py` trainer
- Create `train_all_buckets.py` orchestrator
- Validate all 8 buckets trained successfully

**Estimated Time:** 10-14 hours  
**Risk Level:** MEDIUM (parallel training may exhaust Mac M4 resources)

**Output:** 8 bucket specialists + 1 main predictor + 1 Chronos baseline = 10 L0 models

---

### Phase 4: AutoGluon Stacking & Monte Carlo (7 tasks)
**Status:** NOT STARTED  
**Goal:** Build L1 stacking, L2.5 greedy ensemble, L3 Monte Carlo  
**Key Tasks:**
- Create L1 stacking layer (AutoGluon automatic)
- Create L2.5 greedy weighted ensemble (UPGRADED FEATURE)
- Create L3 Monte Carlo simulation (VaR/CVaR)
- Create full pipeline orchestrator (L0 → L1 → L2 → L2.5 → L3)
- Validate quantile calibration (~10% below Q10, ~10% above Q90)

**Estimated Time:** 12-16 hours  
**Risk Level:** HIGH (ensemble optimization complexity)

**Architecture:**
- **L0:** 10 models (8 buckets + main + Chronos)
- **L1:** AutoGluon stacking layer (trains on L0 OOF predictions)
- **L2:** WeightedEnsemble_L2 (automatically created by AutoGluon)
- **L2.5:** Greedy weighted ensemble (UPGRADED FEATURE - user explicitly wants this)
- **L3:** Monte Carlo simulation (10K scenarios → VaR/CVaR)

---

### Phase 5: Trigger.dev Orchestration (7 tasks)
**Status:** NOT STARTED  
**Goal:** Automate daily training, forecasting, and monitoring  
**Key Tasks:**
- Create daily training Trigger job (2 AM UTC)
- Create daily forecast generation job (10 AM UTC)
- Create model performance monitoring job (6 PM UTC)
- Create weekly retraining job (Sunday midnight)
- Create data quality monitoring job (hourly)
- Create notification system (Slack/email)

**Estimated Time:** 8-12 hours  
**Risk Level:** MEDIUM (Trigger.dev deployment complexity)

**Trigger.dev Jobs:**
1. `daily-training` — Full pipeline training (2 AM UTC)
2. `daily-forecast` — Generate forecasts (10 AM UTC)
3. `model-monitoring` — Performance + drift detection (6 PM UTC)
4. `weekly-retraining` — Full retraining (Sunday midnight)
5. `data-quality-monitoring` — 6-dimension checks (hourly)

---

## ⏱️ TOTAL ESTIMATED TIME

| Phase | Tasks | Hours | Risk |
|-------|-------|-------|------|
| Phase -1 | 6 | 1 | LOW |
| Phase 0 | 13 | 8-12 | HIGH |
| Phase 1 | 7 | 12-16 | MEDIUM |
| Phase 2 | 9 | 8-10 | HIGH |
| Phase 3 | 5 | 10-14 | MEDIUM |
| Phase 4 | 7 | 12-16 | HIGH |
| Phase 5 | 7 | 8-12 | MEDIUM |
| **TOTAL** | **49** | **59-81** | **HIGH** |

**Estimated Calendar Time:** 2-3 weeks (assuming 4-6 hours/day)

---

## 🚨 CRITICAL SUCCESS FACTORS

1. **Phase 0 MUST complete first** — All downstream work depends on working data pipelines
2. **Mac M4 compatibility** — AutoGluon libomp issues must be resolved
3. **Data quality** — No mock/placeholder data allowed
4. **Big 8 coverage** — All 8 buckets must have required data sources
5. **Quantile calibration** — ~10% below Q10, ~10% above Q90 (validates probabilistic forecasts)
6. **Greedy ensemble** — User explicitly wants L2.5 greedy ensemble as "UPGRADED FEATURE"

---

## 📖 NEXT STEPS

1. **Read all 3 plan documents** in order (Phase 0-1, Phase 2, Phase 3-5)
2. **Execute Phase -1** (repo scaffolding)
3. **Execute Phase 0** (critical infrastructure)
4. **Validate Phase 0** before proceeding
5. **Execute Phases 1-5** sequentially
6. **Final validation** using checklist in Phase 3-5 document

---

**🎯 GOAL:** Production-ready AutoGluon hybrid forecasting system for ZL (soybean oil) futures


