# V15 Locked Features - Complete Inventory

**Date**: November 28, 2025  
**Status**: ✅ **LOCKED** - Ready for BigQuery Implementation

---

## ✅ Locked Feature Inventory

### 1. Technical Indicators (19 features) ✅

**Source**: US Oil Solutions Spec (GS/JPM/Vanguard validated)

| Feature | Formula | Status |
|---------|---------|--------|
| `dist_ema_5` | `(Price / EMA_5) - 1` | ✅ Locked |
| `dist_ema_10` | `(Price / EMA_10) - 1` | ✅ Locked |
| `dist_ema_21` | `(Price / EMA_21) - 1` | ✅ Locked |
| `dist_sma_63` | `(Price / SMA_63) - 1` | ✅ Locked |
| `dist_sma_200` | `(Price / SMA_200) - 1` | ✅ Locked |
| `bb_pct_b` | `(Price - Lower) / (Upper - Lower)` | ✅ Locked |
| `bb_bandwidth` | `(Upper - Lower) / MA` | ✅ Locked |
| `ppo_12_26` | `(EMA_12 - EMA_26) / EMA_26 * 100` | ✅ Locked |
| `dist_vwap_21d` | `(Close / VWAP_21) - 1` | ✅ Locked |
| `vol_garman_klass_annualized` | Garman-Klass formula × √252 | ✅ Locked |
| `vol_parkinson_annualized` | Parkinson formula × √252 | ✅ Locked |
| `vol_21d` | `STDDEV(Returns) × √252` | ✅ Locked |
| `amihud_illiquidity` | `ABS(Return) / (Volume × Price)` | ✅ Locked |
| `oi_volume_ratio` | `Open_Interest / Volume` | ✅ Locked |
| `boho_spread` | `(ZL/100 × 7.5) - HO` | ✅ Locked |
| `corr_zl_brl_60d` | `CORR(ZL_Returns, BRL_Returns, 60d)` | ✅ Locked |
| `terms_of_trade_zl_brl` | `ZL_Price / BRL_Price` | ✅ Locked |
| `doy_sin` | `SIN(2π × DayOfYear / 365)` | ✅ Locked |
| `doy_cos` | `COS(2π × DayOfYear / 365)` | ✅ Locked |

**File**: `dataform/definitions/03_features/technical_indicators_us_oil_solutions.sqlx`

---

### 2. FX Indicators (16 features) ✅

**Source**: GS/JPM/Hedge Fund aligned

| Feature | Formula | Status |
|---------|---------|--------|
| `brl_momentum_21d` | `(BRL_t / BRL_{t-21}) - 1` | ✅ Locked |
| `brl_momentum_63d` | `(BRL_t / BRL_{t-63}) - 1` | ✅ Locked |
| `brl_momentum_252d` | `(BRL_t / BRL_{t-252}) - 1` | ✅ Locked |
| `dxy_momentum_21d` | `(DXY_t / DXY_{t-21}) - 1` | ✅ Locked |
| `dxy_momentum_63d` | `(DXY_t / DXY_{t-63}) - 1` | ✅ Locked |
| `dxy_momentum_252d` | `(DXY_t / DXY_{t-252}) - 1` | ✅ Locked |
| `brl_volatility_21d` | `STDDEV(BRL_Returns, 21d) × √252` | ✅ Locked |
| `brl_volatility_63d` | `STDDEV(BRL_Returns, 63d) × √252` | ✅ Locked |
| `corr_zl_brl_30d` | `CORR(ZL_Returns, BRL_Returns, 30d)` | ✅ Locked |
| `corr_zl_brl_60d` | `CORR(ZL_Returns, BRL_Returns, 60d)` | ✅ Locked |
| `corr_zl_brl_90d` | `CORR(ZL_Returns, BRL_Returns, 90d)` | ✅ Locked |
| `corr_zl_dxy_30d` | `CORR(ZL_Returns, DXY_Returns, 30d)` | ✅ Locked |
| `corr_zl_dxy_60d` | `CORR(ZL_Returns, DXY_Returns, 60d)` | ✅ Locked |
| `corr_zl_dxy_90d` | `CORR(ZL_Returns, DXY_Returns, 90d)` | ✅ Locked |
| `corr_regime_zl_brl` | Classification (high/medium/low) | ✅ Locked |
| `corr_regime_zl_dxy` | Classification (high/medium/low) | ✅ Locked |

**File**: `dataform/definitions/03_features/fx_indicators_daily.sqlx`

---

### 3. Fundamental Spreads (5 features) ✅

**Source**: US Oil Solutions Spec (Academic & Institutionally validated)

| Feature | Formula | Status |
|---------|---------|--------|
| `board_crush` | `(ZM × 0.022 + ZL × 11) - ZS` | ✅ Locked |
| `oil_share` | `(ZL × 11) / Board_Crush_Value` | ✅ Locked |
| `hog_spread_feeder_margin` | `HE - (0.8 × ZC + 0.2 × ZM)` | ✅ Locked |
| `boho_spread_gal` | `(ZL/100 × 7.5) - HO` | ✅ Locked |
| `china_pulse_corr_60d` | `CORR(HG_Returns, ZS_Returns, 60d)` | ✅ Locked |

**File**: `dataform/definitions/03_features/fundamental_spreads_daily.sqlx`

---

### 4. Pair Correlations (112 features) ✅

**Source**: All symbol pairs, 4 horizons

| Pairs | Horizons | Total Features | Status |
|-------|----------|----------------|--------|
| 28 pairs (8 choose 2) | 30d, 60d, 90d, 252d | 112 features | ✅ Locked |

**Symbols**: ZL, ZS, ZM, CL, HO, FCPO, 6L, DX

**File**: `dataform/definitions/03_features/pair_correlations_daily.sqlx`

---

### 5. Cross-Asset Betas (28 features) ✅

**Source**: ZL beta vs all other assets, 4 horizons

| Assets | Horizons | Total Features | Status |
|--------|----------|----------------|--------|
| 7 assets (ZS, ZM, CL, HO, FCPO, BRL, DXY) | 30d, 60d, 90d, 252d | 28 features | ✅ Locked |

**Formula**: `COV(ZL, Asset) / VAR(Asset)`

**File**: `dataform/definitions/03_features/cross_asset_betas_daily.sqlx`

---

### 6. Lagged Features (96 features) ✅

**Source**: Prices, returns, indicators, 6 lags

| Symbols | Lags | Features per Symbol | Total Features | Status |
|---------|------|---------------------|----------------|--------|
| 8 symbols | 1d, 2d, 3d, 5d, 10d, 21d | 12 lags | 96 features | ✅ Locked |

**File**: `dataform/definitions/03_features/lagged_features_daily.sqlx`

---

## 📊 Total Locked Features

| Category | Features | Status |
|----------|----------|--------|
| **Technical Indicators** | 19 | ✅ Locked |
| **FX Indicators** | 16 | ✅ Locked |
| **Fundamental Spreads** | 5 | ✅ Locked |
| **Pair Correlations** | 112 | ✅ Locked |
| **Cross-Asset Betas** | 28 | ✅ Locked |
| **Lagged Features** | 96 | ✅ Locked |
| **TOTAL** | **276 features** | ✅ **LOCKED** |

---

## 🎯 Symbols Locked In

### Commodities (8 symbols)
- ✅ ZL (Soybean Oil) - PRIMARY TARGET
- ✅ ZS (Soybeans)
- ✅ ZM (Soybean Meal)
- ✅ CL (Crude Oil)
- ✅ HO (Heating Oil)
- ✅ FCPO (Palm Oil)
- ✅ ZC (Corn)
- ✅ HE (Lean Hogs)

### FX (2 symbols)
- ✅ 6L (BRL Futures)
- ✅ DX (DXY Futures)

### Optional (2 symbols)
- ⚠️ HG (Copper) - For China Pulse
- ⚠️ GC (Gold) - For Real-Terms Price

**Total**: 10-12 symbols ✅

---

## 📋 Prerequisites Before Baselines

### Must Complete (Before Baseline Training)

1. ✅ **USDA Ingestion**
   - WASDE reports
   - Crop Progress
   - Export Sales Reports
   - **Status**: ⚠️ **REQUIRED**

2. ✅ **CFTC Ingestion**
   - COT positions
   - Managed Money positions (ZL-specific)
   - **Status**: ⚠️ **REQUIRED**

3. ✅ **EIA Ingestion**
   - D4/D6 RIN prices
   - Biodiesel production
   - RFS mandate volumes
   - **Status**: ⚠️ **REQUIRED**

---

## ✅ BigQuery Structure Requirements

### Partitioning Strategy
- ✅ All tables: `PARTITION BY DATE(date)`
- ✅ Clustering: `CLUSTER BY symbol` (where applicable)

### No Joins in Skeleton
- ✅ Each table is independent
- ✅ Joins happen in `daily_ml_matrix` (master join table)
- ✅ Skeleton structure = table definitions only

---

**Last Updated**: November 28, 2025

