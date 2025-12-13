# V15 Complete Recovery & Implementation Plan
**Status:** FORENSIC REVIEW COMPLETE — Ready for Phase 0 execution  
**Date:** 2025-12-12  
**Total Tasks:** 60+ across 6 phases  
**Critical Path:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

---

## EXECUTIVE SUMMARY

Forensic review identified:
- **Symbol discrepancy:** 38 symbols in code, but only 33 are CME/CBOT/NYMEX/COMEX (remove KE, MW, UL, AL, TY)
- **Missing reference schema:** 7 tables (regime_calendar, splits, neural_drivers, etc.) — CRITICAL
- **Incomplete features:** ~50% of 276 features implemented (missing pair correlations, betas, lagged features)
- **Missing data flow:** EPA RIN collector, USDA verification, news bucket segmentation at ingestion
- **Missing tables:** 8 feature tables, 5 staging tables, 2 training tables, 2 ops tables

---

## PHASE 0: CRITICAL FIXES (P0 - Blocks Everything)

**Task 0.1:** Fix symbol lists (33 symbols only)
- Files: `trigger/DataBento/Scripts/collect_daily.py`, `databento_ingest_job.ts`
- Action: Remove KE, MW, UL, AL, TY

**Task 0.2:** Create reference schema DDL
- File: `database/models/00_init/01_reference_tables.sql` (NEW)
- Tables: trading_calendar, regime_calendar, regime_weights, train_val_test_splits, feature_catalog, model_registry, neural_drivers

**Task 0.3:** Create reference seed data
- File: `scripts/setup/initialize_reference_tables.sql` (UPDATE)
- Seed: Regime dates, split dates, neural driver mappings

**Task 0.4:** Fix table references
- `database/macros/big8_bucket_features.sql`: fred_observations → fred_economic, eia_petroleum → eia_biofuels

**Task 0.5:** Validate Phase 0
- Run: `python scripts/setup_database.py --both`
- Verify: 33 symbols, reference tables exist with seed data

---

## PHASE 1: FEATURE IMPLEMENTATION (P1)

**Task 1.1-1.5:** Complete 276 features
- Pair correlations: 112 features (28 pairs × 4 horizons)
- Cross-asset betas: 28 features (7 assets × 4 horizons)
- Lagged features: 96 features (8 symbols × 12 lags)
- FX indicators: 16 features (momentum, vol, correlation)
- Technical indicators: 19 features (verify all present)

**Task 1.6-1.10:** Create missing tables
- `features.sentiment_features_daily` (14 features)
- `features.trump_news_features_daily` (6-10 features)
- `features.neural_signals_daily` (Layer 2 scores)
- `staging.sentiment_buckets` (aggregated sentiment)
- `staging.fred_macro_clean` (cleaned FRED data)

---

## PHASE 2: DATA FLOW RECOVERY (P1)

**Task 2.1:** News bucket segmentation at ingestion
- File: `trigger/ScrapeCreators/Scripts/collect_news_buckets.py` (UPDATE)
- Add: Bucket assignment + FinBERT sentiment at ingestion

**Task 2.2:** Create EPA RIN prices collector
- File: `trigger/EPA/Scripts/collect_rin_prices.py` (NEW)
- Data: D3, D4, D5, D6 RIN prices (weekly)

**Task 2.3:** Verify USDA collectors
- Files: `trigger/USDA/Scripts/collect_*.py`
- Verify: WASDE, export sales, crop progress

---

## PHASE 3: AUTOGLUON INTEGRATION (P1)

**Task 3.1:** Install AutoGluon 1.4 on Mac M4
- Script: `scripts/setup/install_autogluon_mac.sh`
- Fix: libomp compatibility

**Task 3.2:** Create AutoGluon wrappers
- `src/training/autogluon/tabular_trainer.py` (TabularPredictor)
- `src/training/autogluon/timeseries_trainer.py` (TimeSeriesPredictor)
- `src/training/autogluon/bucket_specialist.py` (bucket training)

**Task 3.3:** Train 8 bucket specialists
- Config: `config/training/buckets/*.yaml`
- Orchestrator: `src/training/autogluon/train_all_buckets.py`

---

## PHASE 4: ENSEMBLE & MONTE CARLO (P2)

**Task 4.1:** Extract AutoGluon ensemble weights
- File: `src/training/autogluon/extract_ensemble_weights.py`
- Save to: `training.ensemble_weights`

**Task 4.2:** Update QRA ensemble for AutoGluon outputs
- File: `src/ensemble/qra_ensemble.py` (UPDATE)

**Task 4.3:** Update Monte Carlo for quantile inputs
- File: `src/simulators/monte_carlo_sim.py` (UPDATE)

---

## PHASE 5: TRIGGER.DEV ORCHESTRATION (P2)

**Task 5.1:** Create training orchestration job
- File: `trigger/autogluon_training_orchestrator.ts`
- Schedule: Weekly Sunday 2 AM UTC

**Task 5.2:** Create daily forecast job
- File: `trigger/autogluon_daily_forecast.ts`
- Schedule: Daily 6 PM UTC

**Task 5.3:** Create model monitoring job
- File: `trigger/autogluon_model_monitoring.ts`
- Schedule: Daily 8 PM UTC

---

## KEY PATTERNS TO PRESERVE

1. **Denormalization:** No JOINs at runtime, all features pre-computed
2. **Idempotency:** INSERT OR REPLACE, date-based partitioning
3. **Segmentation at Ingestion:** News buckets assigned immediately (Python)
4. **Single Source of Truth:** Feature formulas locked in SQL macros

---

## VALIDATION CHECKPOINTS

- **After Phase 0:** Schemas created, 33 symbols verified, reference data seeded
- **After Phase 1:** All 276 features computable, missing tables created
- **After Phase 2:** Data flows Raw → Staging → Features, news segmented at ingestion
- **After Phase 3:** 8 bucket specialists trained, OOF predictions saved
- **After Phase 4:** Ensemble weights learned, Monte Carlo scenarios generated
- **After Phase 5:** End-to-end pipeline automated, forecasts on dashboard

---

**Next Step:** Execute Phase 0 (symbol fix + reference schema)

