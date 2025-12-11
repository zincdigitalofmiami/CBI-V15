# Phase 0 Execution Ready - AutoGluon Big 8 Ensemble

**Date**: December 9, 2025  
**Status**: ✅ CLEARED FOR EXECUTION  
**Plan**: `/Users/zincdigital/.cursor/plans/autogluon_hybrid_implementation_c2287cb0.plan.md` (882 lines)

---

## ✅ Architecture Verified

### Primary Goal
Predict probabilistic ZL (soybean oil futures) returns/levels at **1w, 1m, 3m, 6m** horizons using:
- **8 Big 8 bucket specialists** + 1 main ZL predictor
- **AutoGluon 1.4** TabularPredictor (preset="best", quantile regression)
- **Mac M4 CPU** training (NO GPU required)
- **MotherDuck** as source of truth (permanent storage)
- **Local Mac DuckDB** as persistent training workspace (100-1000x faster I/O)

### Model Stack (4 Layers)

**L0**: 9 TabularPredictors (8 Big 8 + 1 main)
- Each trains 10-15 models (LightGBM, CatBoost, XGBoost, Neural Nets)
- Each creates WeightedEnsemble_L2 automatically
- Outputs: OOF predictions (P10, P50, P90)

**L1**: Meta-Learner (ensemble of 9 specialists)
- AutoGluon learns optimal weights from OOF predictions
- Final output: P10, P50, P90 for ZL

**L2**: Production Forecasts
- Upload to MotherDuck: `forecasts.zl_predictions`
- Dashboard queries this table

**L3**: Monte Carlo Simulation (Risk Metrics ONLY)
- Input: L2 predictions
- Purpose: VaR/CVaR calculation (NOT forecasting)
- Location: `src/simulators/monte_carlo_sim.py`

---

## 🎯 Big 8 Bucket Specialists

All 8 buckets verified and configured:

| # | Bucket | Core Features | Bucket-Specific | Total | News Feeds |
|---|--------|---------------|-----------------|-------|------------|
| 1 | **Crush** | ~50 | +20 | ~70 | farmdoc Grain Outlook |
| 2 | **China** | ~50 | +15 | ~65 | Farm Policy News (trade), USDA exports |
| 3 | **FX** | ~50 | +10 | ~60 | Multi-currency data feeds |
| 4 | **Fed** | ~50 | +15 | ~65 | Farm Policy News (budget), Fed speeches |
| 5 | **Tariff** | ~50 | +20 | ~70 | Trump Truth Social, Farm Policy News |
| 6 | **Biofuel** | ~50 | +25 | ~75 | farmdoc Daily RIN, EPA RIN prices |
| 7 | **Energy** | ~50 | +20 | ~70 | EIA petroleum reports |
| 8 | **Volatility** | ~50 | +15 | ~65 | VIX, stress indices |
| 9 | **Main ZL** | ALL | ALL | ~300 | ALL feeds combined |

### Feature Architecture (ADDITIVE)

**Layer 1: Core Macro/FX** (~50 features - ALL buckets inherit):
- FX (16): Rate differential, BRL/DXY momentum, volatility, correlations, **Terms of Trade**
- Macro (12): Fed funds, yields, curves, NFCI, STLFSI4, VIX, UNRATE, CPI
- Price/Volume (3): ZL close, volume, OI
- Cross-Asset (5): Board crush, oil share, BOHO, HG proxy, DX

**Layer 2: Bucket-Specific** (10-25 features ADDED per bucket):
- Crush adds: ZL/ZS/ZM spreads, board crush details, hog spread
- China adds: USDA export sales, HG-ZS correlation, Farm Policy News
- FX adds: CNY/MXN pairs, multi-pair correlations
- Fed adds: Policy indicators, FOMC analysis
- Tariff adds: Trump sentiment, Section 301, trade policy
- Biofuel adds: EPA RIN D4/D6 (weekly→daily filled), BOHO, biodiesel
- Energy adds: CL/HO/RB, crack spreads, CL-ZL correlation
- Volatility adds: VIX term structure, realized vol, stress indices

**Main ZL**: ALL ~300 features (no specialization, full 360° view)

---

## ⚠️ Phase 0 Checkpoint Sequence (EXACT ORDER REQUIRED)

**DEPENDENCY GRAPH**: 0.1 → 0.2 → 0.3 → 0.4

### Script 0.1: Install libomp (Mac M4)
```bash
bash scripts/setup/install_autogluon_mac.sh
```
- **Why First**: LightGBM SEGFAULTS without OpenMP (libomp)
- **Blocks**: ALL AutoGluon training
- **Time**: 5-10 minutes
- **Validation**: `python3 -c "from autogluon.tabular import TabularPredictor; print('Ready!')"`

### Script 0.2: Fix date → as_of_date
```bash
# Manual edit: trigger/DataBento/Scripts/collect_daily.py
# Lines 146, 160, 165, 196, 234
# Change: "ts_event": "date" → "ts_event": "as_of_date"
```
- **Why Second**: core_macro_fx view relies on consistent join keys
- **Blocks**: Script 0.3 (view creation will fail if joins break)
- **Time**: 2-5 minutes
- **Validation**: `python trigger/DataBento/Scripts/collect_daily.py --symbol ZL --days 5`

### Script 0.3: Deploy features.core_macro_fx View
```bash
python scripts/setup_database.py --both
```
- **Why Third**: Provides ~50 base features to ALL 8 buckets
- **Blocks**: ALL bucket training (buckets query this view)
- **Time**: 1 minute
- **Validation**: `SELECT COUNT(*) FROM features.core_macro_fx;`

### Script 0.4: Verify Terms of Trade Calculation
```bash
python scripts/validation/verify_core_macro_fx.py
```
- **Why Fourth**: Prevents Inf/NaN crashes (BRL price can be zero)
- **Blocks**: Training stability
- **Time**: 1 minute
- **Validation**: Should print `✅ CORE_MACRO_FX VERIFICATION COMPLETE`

---

## ✅ Files Created in Phase 0.0 (COMPLETED)

1. ✅ `scripts/setup/install_autogluon_mac.sh` - Mac M4 libomp fix (prevents segfaults)
2. ✅ `database/definitions/03_features/core_macro_fx.sql` - ~50 base features view
3. ✅ `scripts/validation/verify_core_macro_fx.py` - Terms of Trade validator
4. ✅ `src/reporting/training_auditor.py` - Hot-audit loop (immediate reporting)
5. ✅ `database/macros/anofox_guards.sql` - SQL data quality guards (fail-fast)
6. ✅ `database/definitions/01_raw/ops_training_logs.sql` - Audit log table
7. ✅ `config/bucket_feature_selectors.yaml` - ADDITIVE feature model (8 buckets + main)
8. ✅ `scripts/sync_motherduck_to_local.py` - MotherDuck → Local sync
9. ✅ `.augment.md` - Augment workspace instructions (289 lines)
10. ✅ Fixed `database/macros/big8_bucket_features.sql` - Weekly→daily fill, table refs

---

## 🚀 Ready for Execution

**Current Status:**
- ✅ Architecture verified (MotherDuck + Mac M4 + AutoGluon 1.4)
- ✅ All 8 Big 8 buckets configured
- ✅ ADDITIVE feature model designed (core + bucket-specific)
- ✅ Critical holes patched (libomp, weekly→daily, Terms of Trade guards)
- ✅ Hot-audit loop created (immediate reporting)
- ✅ Sync script ready (MotherDuck → Local)

**Next Action:**
Execute Phase 0 scripts in EXACT order (0.1 → 0.2 → 0.3 → 0.4)

**Expected Duration:**
- Phase 0: 10-20 minutes (4 scripts)
- Phase 1: 2-4 hours (data ingestion - EPA, CFTC, FRED, News scrapers)
- Phase 2-3: 12-21 hours (AutoGluon training all buckets)
- Phase 4-5: 4-8 hours (ensemble + Trigger.dev orchestration)

**Total**: ~18-33 hours for complete implementation

---

## 📚 Critical References for Augment Code

1. **This plan**: `/Users/zincdigital/.cursor/plans/autogluon_hybrid_implementation_c2287cb0.plan.md`
2. **Architecture**: `docs/architecture/MASTER_PLAN.md`
3. **Guardrails**: `AGENTS.md`
4. **Workspace**: `.augment.md`
5. **Features**: `config/bucket_feature_selectors.yaml`

**All systems GO for Phase 0 execution!** 🚀
