# Baseline v15.0 Strategy - Proceed NOW

**Date**: November 28, 2025  
**Status**: ✅ **READY TO EXECUTE**

---

## 🎯 Strategy: Don't Wait for Perfect

### Key Insight
**We don't need USDA/CFTC/EIA to start baselines. We have 276 features locked and ready.**

### Approach
1. **Baseline v15.0**: Train with current 276 features (what we have NOW)
2. **Baseline v15.1**: Add USDA/CFTC/EIA later, re-run, compare performance

---

## ✅ What We Have NOW (276 Features)

### Ready to Use:
- ✅ Technical Indicators: 19 features
- ✅ FX Indicators: 16 features
- ✅ Fundamental Spreads: 5 features
- ✅ Pair Correlations: 112 features
- ✅ Cross-Asset Betas: 28 features
- ✅ Lagged Features: 96 features

### Missing (Can Add Later):
- ⚠️ USDA Crop Progress (can add to v15.1)
- ⚠️ CFTC Managed Money positions (can add to v15.1)
- ⚠️ EIA RIN prices (can add to v15.1)

**Verdict**: ✅ **276 features is MORE than enough to start baselines**

---

## 📊 Implementation Steps

### Step 1: Build `daily_ml_matrix` ✅

**File**: `dataform/definitions/03_features/daily_ml_matrix.sqlx`

**What it does**:
- Joins all 276 features
- Generates targets (price levels) for 1w, 1m, 3m, 6m
- One row per symbol, date

**Status**: ✅ **READY**

---

### Step 2: Create Train/Val/Test Splits ✅

**Files**:
- `dataform/definitions/04_training/train_val_test_splits.sqlx`
- `dataform/definitions/04_training/daily_ml_matrix_train.sqlx`
- `dataform/definitions/04_training/daily_ml_matrix_val.sqlx`
- `dataform/definitions/04_training/daily_ml_matrix_test.sqlx`

**Splits**:
- Train: 2010-01-01 to 2018-12-31
- Val: 2019-01-01 to 2021-12-31
- Test: 2022-01-01 onwards

**Status**: ✅ **READY**

---

### Step 3: Export Training Data ✅

**Script**: `scripts/export/export_training_data.py`

**What it does**:
- Exports train/val/test splits as Parquet
- Saves to external drive

**Status**: ✅ **READY**

---

### Step 4: Train LightGBM Baselines ✅

**Script**: `src/training/baselines/lightgbm_zl.py`

**What it does**:
- Trains one model per horizon (1w, 1m, 3m, 6m)
- Evaluates on train/val/test splits
- Saves models and predictions

**Status**: ✅ **READY**

---

## 🎯 Baseline v15.0 Goals

### Success Criteria:
1. ✅ Models train without errors
2. ✅ Val MAE < 5% (baseline target)
3. ✅ Test MAE < 5% (baseline target)
4. ✅ R² > 0.80 (baseline target)

### Outputs:
- ✅ 4 LightGBM models (one per horizon)
- ✅ Performance metrics (MAE, RMSE, R²)
- ✅ Predictions on test set
- ✅ Feature importance rankings

---

## 📊 Baseline v15.1 (After USDA/CFTC/EIA)

### What Changes:
- Add USDA features (crop progress, exports)
- Add CFTC features (managed money positions)
- Add EIA features (RIN prices, biodiesel production)
- Rebuild `daily_ml_matrix` with full feature set
- Re-run same baseline script
- Compare performance vs v15.0

### Expected Improvement:
- Val MAE: 5% → 4% (target)
- Test MAE: 5% → 4% (target)
- R²: 0.80 → 0.85 (target)

---

## ✅ Action Plan

### This Week:
1. ✅ Build `daily_ml_matrix` in Dataform
2. ✅ Export training data
3. ✅ Train LightGBM baselines (v15.0)
4. ✅ Document baseline performance

### Next Week:
1. ⚠️ Implement USDA ingestion (parallel)
2. ⚠️ Implement CFTC ingestion (parallel)
3. ⚠️ Implement EIA ingestion (parallel)

### Week After:
1. ⚠️ Rebuild `daily_ml_matrix` with full features
2. ⚠️ Re-run baselines (v15.1)
3. ⚠️ Compare performance vs v15.0

---

## 🎯 Summary

**Strategy**: ✅ **PROCEED WITH BASELINES NOW**

- ✅ We have 276 features ready
- ✅ We have all infrastructure ready
- ✅ We don't need to wait for USDA/CFTC/EIA
- ✅ We can add them later and compare performance

**Next Action**: Build `daily_ml_matrix` and start training!

---

**Last Updated**: November 28, 2025

