# Baseline Training Pipeline - Complete Flow (Historical)

**Date**: November 28, 2025  
**Status**: ✅ **READY FOR IMPLEMENTATION** (for **legacy LightGBM-only baselines**, not the V15.1 AutoGluon stack)

---

## 🎯 Pipeline Overview (Legacy LightGBM Baseline)

### Flow: DuckDB/MotherDuck → Mac → DuckDB/MotherDuck

```
DuckDB/MotherDuck (Pre-Compute)
    ↓
Export Training Data (~365 features)
    ↓
Mac Training (LightGBM)
    ↓
Model Evaluation
    ↓
Upload Predictions
    ↓
DuckDB/MotherDuck (Forecasts)
```

---

## 📊 Step 1: DuckDB/MotherDuck Pre-Compute (DONE)

### Features Pre-Computed (~365 features)

#### Technical Indicators (19 features)

- Distance MAs: 5 features
- Bollinger: 2 features
- PPO: 1 feature
- VWAP: 1 feature
- Volatility: 3 features
- Microstructure: 2 features
- Cross-asset: 3 features
- Metadata: 2 features

#### FX Indicators (16 features)

- BRL Momentum: 3 features
- DXY Momentum: 3 features
- BRL Volatility: 2 features
- ZL-BRL Correlation: 3 features
- ZL-DXY Correlation: 3 features
- Terms of Trade: 1 feature
- Correlation Regimes: 2 features

#### Fundamental Spreads (4 features)

- Board Crush: 1 feature
- Oil Share: 1 feature
- Hog Spread: 1 feature
- BOHO Spread: 1 feature
- China Pulse: 1 feature (optional)

#### Pair Correlations (112 features)

- 28 pairs × 4 horizons = 112 features

#### Cross-Asset Betas (28 features)

- 7 assets × 4 horizons = 28 features

#### Lagged Features (96 features)

- 8 symbols × 12 lags = 96 features

#### Additional Pre-Compute (90 features)

- Rolling statistics: 50 features
- Feature interactions: 20 features
- Factor loadings: 10 features
- Regime indicators: 10 features

**Total**: ~365 features pre-computed in DuckDB/MotherDuck ✅

---

## 📥 Training & Export (Current Guidance)

Use the AutoGluon-based Big 8 + ZL stack with DuckDB/MotherDuck per `docs/architecture/MASTER_PLAN.md`, `PHASE_0_EXECUTION_READY.md`, and `AGENTS.md`. Legacy BigQuery/`google.cloud` export/upload scripts have been removed.

---

## ✅ Pipeline Checklist (AutoGluon/DuckDB/MotherDuck)

- DuckDB/MotherDuck pre-compute: ~365 features
- Training/export: AutoGluon-driven, reading from local DuckDB mirror
- Forecast upload: write to `forecasts.*` in MotherDuck
- Validation: monitor feature quality, splits, and model performance

---

**Last Updated**: December 2025
