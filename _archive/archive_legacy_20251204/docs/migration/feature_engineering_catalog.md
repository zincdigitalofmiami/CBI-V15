# Feature Engineering Catalog - 276 Features

**Date:** December 3, 2024  
**Source:** `src/features/build_daily_ml_matrix.py`  
**Status:** Documentation in progress

---

## Overview

The feature engineering pipeline creates a comprehensive daily ML matrix with 276+ engineered features from multiple data sources. This document catalogs all features, their dependencies, and calculation formulas.

---

## Feature Categories

### 1. Market Data Features (Base)
**Source:** `staging.market_daily`

- `date` - Date key
- `symbol` - Symbol (ZL, HO, etc.)
- `open`, `high`, `low`, `close` - OHLC prices
- `volume` - Trading volume
- `price` - Close price (alias)

**Dependencies:** Raw Databento futures data

---

### 2. Technical Indicators (19 features)
**Source:** Computed from market data

- `rsi_14` - Relative Strength Index (14-day)
- `macd` - MACD line
- `macd_signal` - MACD signal line
- `macd_histogram` - MACD histogram
- `sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_200` - Simple Moving Averages
- `ema_12`, `ema_26` - Exponential Moving Averages
- `atr_14` - Average True Range (14-day)
- `bb_upper`, `bb_middle`, `bb_lower` - Bollinger Bands
- `gk_vol_21d` - Garman-Klass volatility (21-day, annualized)

**Formula - Garman-Klass:**
```
GK = sqrt(0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2)
Annualized: GK_21d * sqrt(252)
```

**Dependencies:** `staging.market_daily`

---

### 3. FX Technical Indicators (16 features)
**Source:** `raw.fred_economic` (DEXBZUS, DTWEXBGS)

**BRL (Brazilian Real) Features:**
- `fx_brl_mom_21d` - 21-day momentum
- `fx_brl_mom_63d` - 63-day momentum
- `fx_brl_mom_252d` - 252-day momentum
- `fx_brl_vol_21d` - 21-day volatility (annualized)
- `fx_brl_vol_63d` - 63-day volatility (annualized)

**DXY (Dollar Index) Features:**
- `fx_dxy_mom_21d` - 21-day momentum
- `fx_dxy_mom_63d` - 63-day momentum
- `fx_dxy_mom_252d` - 252-day momentum
- `fx_dxy_vol_21d` - 21-day volatility (annualized)
- `fx_dxy_vol_63d` - 63-day volatility (annualized)

**Cross-Asset Correlations:**
- `corr_zl_brl_30d` - ZL-BRL 30-day correlation
- `corr_zl_brl_60d` - ZL-BRL 60-day correlation
- `corr_zl_brl_90d` - ZL-BRL 90-day correlation
- `corr_zl_dxy_30d` - ZL-DXY 30-day correlation
- `corr_zl_dxy_60d` - ZL-DXY 60-day correlation
- `corr_zl_dxy_90d` - ZL-DXY 90-day correlation

**Terms of Trade:**
- `terms_of_trade_zl_brl` - ZL price / BRL FX level

**Dependencies:** `raw.fred_economic`

---

### 4. FRED Macro Levels (20+ features)
**Source:** `raw.fred_economic`

**Policy Rates:**
- `fred_dff` - Fed Funds Rate
- `fred_dfedtaru` - Fed Funds Upper Target
- `fred_dfedtarl` - Fed Funds Lower Target
- `fred_effr` - Effective Fed Funds Rate
- `fred_sofr` - SOFR

**Treasury Curve:**
- `fred_dgs3mo` - 3-Month Treasury
- `fred_dgs1` - 1-Year Treasury
- `fred_dgs2` - 2-Year Treasury
- `fred_dgs5` - 5-Year Treasury
- `fred_dgs10` - 10-Year Treasury
- `fred_dgs30` - 30-Year Treasury

**Spreads:**
- `fred_t10y2y` - 10Y-2Y Spread
- `fred_t10y3m` - 10Y-3M Spread

**Financial Conditions:**
- `fred_vixcls` - VIX Close
- `fred_nfci` - National Financial Conditions Index
- `fred_nfcileverage` - NFCI Leverage Component
- `fred_baaffm` - BAA Corporate Bond Yield
- `fred_bamlh0a0hym2` - High Yield Spread

**Dependencies:** `raw.fred_economic`

---

### 5. Fundamental Spreads (5 features)
**Source:** Computed from market data

**BOHO Spread (LOCKED FORMULA):**
- `boho_spread` - (ZL/100 * 7.5) - HO
  - ZL in cents/lb, HO in $/gal
  - Conversion: 7.5 lbs per gallon
  - Positive = ZL expensive, Negative = HO expensive

**Dependencies:** `staging.market_daily` (ZL, HO symbols)

---

### 6. Lagged Features (96 features)
**Source:** Computed from base features

Lagged versions of key features at multiple horizons:
- Price lags: 1d, 2d, 3d, 5d, 10d, 21d, 63d
- Volume lags: 1d, 2d, 3d, 5d, 10d
- RSI lags: 1d, 2d, 3d, 5d
- MACD lags: 1d, 2d, 3d
- FX momentum lags: 1d, 2d, 3d, 5d
- VIX lags: 1d, 2d, 3d, 5d, 10d

**Dependencies:** All base features

---

### 7. Pair Correlations (112 features)
**Source:** Computed from multiple symbols

Rolling correlations between:
- ZL vs other commodities (HO, CL, etc.)
- ZL vs FX pairs (BRL, DXY)
- ZL vs macro indicators (VIX, rates)
- Cross-commodity correlations

**Windows:** 30d, 60d, 90d, 180d

**Dependencies:** `staging.market_daily`, `raw.fred_economic`

---

### 8. Cross-Asset Betas (28 features)
**Source:** Computed from price relationships

Beta coefficients measuring ZL sensitivity to:
- Crude oil (CL)
- Heating oil (HO)
- Gasoline (RB)
- FX pairs (BRL, DXY)
- Macro indicators (VIX, rates)

**Windows:** 30d, 60d, 90d

**Dependencies:** `staging.market_daily`, `raw.fred_economic`

---

## Total Feature Count

**Approximate breakdown:**
- Market data base: 7 features
- Technical indicators: 19 features
- FX technicals: 16 features
- FRED macro: 20+ features
- Fundamental spreads: 5 features
- Lagged features: 96 features
- Pair correlations: 112 features
- Cross-asset betas: 28 features
- **Total: ~276 features**

---

## Dependencies Map

### Raw Data Sources:
1. `raw.databento_futures_ohlcv_1d` → Market base features
2. `raw.databento_futures_ohlcv_1h` → Intraday features (if used)
3. `raw.fred_economic` → FX and macro features

### Staging Tables:
1. `staging.market_daily` → Base market features
2. `staging.fred_macro_panel` → Macro features (if exists)

### Reference Tables:
1. `reference.regime_calendar` → Regime weights
2. `reference.regime_weights` → Regime-specific weights

---

## Migration to DuckDB

**Status:** In Progress

**Changes Required:**
1. Replace `load_table()` calls with `duckdb_utils.load_table_from_duckdb()`
2. Update SQL queries to use DuckDB syntax (mostly compatible)
3. Test feature pipeline end-to-end with DuckDB data

**Files to Update:**
- `src/features/build_daily_ml_matrix.py` - Main feature builder
- `src/features/build_fx_indicators_daily.py` - FX features
- `src/features/technical_indicators_bigquery.py` - Technical indicators

---

## Notes

- All features are computed daily (no intraday features in base matrix)
- Features use forward-fill for missing values (weekends/holidays)
- Regime weights are applied during training, not in feature matrix
- Feature order is preserved for model compatibility

---

**Last Updated:** December 3, 2024

