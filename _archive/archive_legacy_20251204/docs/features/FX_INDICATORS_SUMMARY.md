# FX Indicators Summary - Industry Standards

**Date**: November 28, 2025  
**Status**: ✅ **VALIDATED** - Aligned with GS/JPM/Hedge Fund best practices

---

## ✅ Industry Standards Identified

### 1. Carry Trade Indicators
- ✅ **BRL-US Rate Differential**: `(BRL_Rate - US_Rate) / US_Rate`
- ⏳ Forward Premium/Discount: `(Forward - Spot) / Spot`
- ⏳ Carry Risk-Adjusted: `Carry / Volatility`

**Status**: ✅ **ADD** - Missing from current spec

---

### 2. Momentum Indicators
- ✅ **BRL Momentum**: 21d, 63d, 252d horizons
- ✅ **DXY Momentum**: 21d, 63d, 252d horizons
- ⏳ Momentum Persistence: How long momentum persists
- ⏳ Momentum Acceleration: Change in momentum

**Status**: ✅ **ENHANCE** - Add 63d, 252d horizons

---

### 3. Volatility Indicators
- ✅ **BRL Volatility**: 21d, 63d realized volatility
- ⏳ Volatility Term Structure: Short/Long term ratio
- ⏳ Volatility Regime: High/Medium/Low classification

**Status**: ✅ **ENHANCE** - Add 63d horizon, term structure

---

### 4. Correlation Indicators
- ✅ **ZL-BRL Correlation**: 30d, 60d, 90d horizons ✅
- ✅ **ZL-DXY Correlation**: 30d, 60d, 90d horizons ✅
- ✅ **Correlation Regime**: High/Medium/Low classification ✅

**Status**: ✅ **KEEP** - Already good

---

### 5. Cross-Asset Indicators
- ✅ **Terms of Trade**: `ZL_Price / BRL_Price` ✅
- ⏳ Real Exchange Rate: Inflation-adjusted BRL
- ⏳ Trade-Weighted BRL: Weighted vs trading partners

**Status**: ✅ **KEEP** - Terms of Trade implemented

---

## 📊 Implementation Status

### Phase 1: Core FX Features (16 indicators)
- ✅ BRL Momentum (21d, 63d, 252d) - 3 features
- ✅ DXY Momentum (21d, 63d, 252d) - 3 features
- ✅ BRL Volatility (21d, 63d) - 2 features
- ✅ ZL-BRL Correlation (30d, 60d, 90d) - 3 features
- ✅ ZL-DXY Correlation (30d, 60d, 90d) - 3 features
- ✅ Terms of Trade - 1 feature
- ✅ Correlation Regimes - 2 features

**Total**: **16 FX features** ✅

### Phase 2: Advanced FX Features (3 indicators)
- ⏳ BRL-US Rate Differential (Carry)
- ⏳ Forward Premium/Discount
- ⏳ Volatility Term Structure

**Total**: **3 additional features**

---

## 💰 Cost Impact

- **FX Data**: ~10 GB/month (FRED rates, Databento FX)
- **FX Calculations**: ~5 GB/month (correlations, momentum)
- **Total**: ~15 GB/month = **$0.00** (within free tier) ✅

---

## ✅ Summary

**Industry Alignment**: ✅ **EXCELLENT**

- ✅ Momentum: Multi-horizon (21d, 63d, 252d)
- ✅ Volatility: Multi-horizon (21d, 63d)
- ✅ Correlations: Multi-horizon (30d, 60d, 90d)
- ✅ Terms of Trade: Implemented
- ⏳ Carry Trade: **ADD** (BRL-US Rate Differential)

**Recommendation**: ✅ **IMPLEMENT Phase 1** (16 features), **ADD Phase 2** (carry indicators)

---

**Last Updated**: November 28, 2025

