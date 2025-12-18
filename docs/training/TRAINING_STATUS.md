# Training Status - Quick Reference

**Last Updated:** December 16, 2025  
**Status:** 🟡 READY FOR TEST RUN

---

## Can We Train? YES ✅ (with limitations)

### What We Have ✅

| Component | Status | Details |
|-----------|--------|---------|
| **ZL Price Data** | ✅ EXCELLENT | 4,017 days (2010-2025), $25-$87 range |
| **FRED Macro** | ✅ EXCELLENT | 252,655 rows, 58 indicators |
| **CFTC Positioning** | ✅ GOOD | 4,506 rows, 16 symbols, 5+ years |
| **USDA Data** | ✅ GOOD | Export sales + WASDE, 5+ years |
| **SQL Macros** | ✅ COMPLETE | 93+ features defined |
| **Training Scripts** | ✅ EXISTS | LightGBM/XGBoost/CatBoost baselines |
| **Schemas** | ✅ DEPLOYED | All 13 schemas in MotherDuck |

### What's Missing ⚠️

| Component | Status | Impact | Priority |
|-----------|--------|--------|----------|
| **Feature Tables** | ❌ EMPTY | Cannot train yet | 🔴 CRITICAL |
| **EPA RIN Prices** | ⚠️ PARTIAL | Biofuel bucket weak | 🟡 HIGH |
| **News Sentiment** | ❌ MISSING | Tariff bucket weak | 🟢 MEDIUM |
| **Orchestration** | ❌ MISSING | No ensemble yet | 🟡 HIGH |

---

## Quick Start (45 minutes)

```bash
# Run this to start test training:
bash scripts/training/quick_test_run.sh
```

**What it does:**
1. Builds all 93+ features from SQL macros (10 min)
2. Syncs MotherDuck → Local DuckDB (5 min)
3. Trains LightGBM baseline model (30 min)

**Expected output:**
- Model artifact: `data/models/lightgbm_zl_baseline.pkl`
- Validation MAPE: < 10%
- Predictions: 1-week and 1-month horizons

---

## Data Coverage Summary

### Raw Data Tables (8 populated)

```
✅ databento_futures_ohlcv_1d    218,941 rows  (2010-2025)
✅ fred_economic                 252,655 rows  (2000-2025)
✅ cftc_cot                        4,506 rows  (2020-2025)
✅ usda_export_sales               6,412 rows  (2020-2025)
✅ usda_wasde                      4,320 rows  (2020-2025)
⚠️ epa_rin_prices                   208 rows  (2024-2025) ⚠️ ONLY 3 WEEKS
✅ weather_noaa                     600 rows  (2024-2025)
❌ scrapecreators_news_buckets       16 rows  (TEST DATA)
```

### Feature Tables (0 populated)

```
❌ daily_ml_matrix_zl              0 rows  ← MASTER TABLE (CRITICAL)
❌ technical_indicators_all_symbols 0 rows
❌ bucket_scores                   0 rows
❌ bucket_crush                    0 rows
❌ bucket_china                    0 rows
❌ bucket_fx                       0 rows
❌ bucket_fed                      0 rows
❌ bucket_tariff                   0 rows
❌ bucket_biofuel                  0 rows
❌ bucket_energy                   0 rows
❌ bucket_volatility               0 rows
```

**Action:** Run `python src/engines/anofox/build_all_features.py`

---

## Training Options

### Option A: Quick Test (TODAY) ✅

**What:** Train single LightGBM model with full features

**Command:**
```bash
bash scripts/training/quick_test_run.sh
```

**Time:** 45 minutes  
**Output:** Baseline model + metrics  
**Limitations:** No ensemble, EPA RIN data incomplete

### Option B: Full V15.1 (NEXT WEEK)

**What:** 3-stage ensemble (8 bucket specialists + core + meta)

**Requirements:**
1. ✅ Complete Option A first
2. ⚠️ Backfill EPA RIN prices (2010-2024)
3. ❌ Create orchestration scripts
4. 🟢 Optional: Add news sentiment

**Time:** 7 hours  
**Output:** Full ensemble with P10/P50/P90 forecasts

---

## Critical Gaps to Address

### 🔴 BLOCKER 1: Feature Engineering (10 minutes)

**Issue:** All feature tables empty  
**Fix:** `python src/engines/anofox/build_all_features.py`  
**Impact:** Cannot train without features

### 🟡 BLOCKER 2: EPA RIN Prices (2-4 hours)

**Issue:** Only 3 weeks of data (need 15 years)  
**Fix:** Create `src/ingestion/eia_epa/backfill_epa_rin_prices.py`  
**Impact:** Biofuel bucket specialist will be weak

### 🟡 BLOCKER 3: Orchestration Scripts (4-6 hours)

**Issue:** No scripts to train bucket specialists + meta model  
**Fix:** Create 4 new training scripts  
**Impact:** Cannot run full V15.1 ensemble

### 🟢 OPTIONAL: News Sentiment (8-12 hours)

**Issue:** No Trump posts or Farm Policy News  
**Fix:** Deploy news scrapers  
**Impact:** Tariff bucket will be weaker (but not critical)

---

## Success Criteria

### Test Run (Today)
- ✅ Features populate successfully
- ✅ Model trains without errors
- ✅ MAPE < 10% on validation
- ✅ Predictions look reasonable

### Full V15.1 (Week 1)
- ✅ All 9 specialists trained
- ✅ Ensemble beats individual models
- ✅ Forecasts in MotherDuck
- ✅ Dashboard displays predictions

---

## File Locations

### Documentation
- **Full audit:** `docs/ops/TRAINING_READINESS_AUDIT.md`
- **This file:** `TRAINING_STATUS.md`

### Scripts
- **Test run:** `scripts/training/quick_test_run.sh`
- **Feature builder:** `src/engines/anofox/build_all_features.py`
- **Sync script:** `scripts/sync_motherduck_to_local.py`
- **Baseline trainer:** `src/training/baselines/lightgbm_zl.py`

### Data
- **MotherDuck:** `md:cbi_v15` (source of truth)
- **Local mirror:** `data/duckdb/cbi_v15.duckdb` (training)
- **Models:** `data/models/` (artifacts)

---

## Next Steps

1. **NOW:** Run `bash scripts/training/quick_test_run.sh`
2. **Tomorrow:** Backfill EPA RIN prices
3. **This week:** Create orchestration scripts
4. **Next week:** Full V15.1 training run

---

**Questions?** See `docs/ops/TRAINING_READINESS_AUDIT.md` for detailed analysis.
