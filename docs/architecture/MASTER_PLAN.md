# MASTER PLAN

**Date:** December 9, 2025  
**Last Revised:** December 9, 2025  
**Status:** V15.1 - AutoGluon Hybrid Architecture  
**Purpose:** Single source of truth for CBI-V15 architecture

---

## PHILOSOPHY

**"SQL-first features, AutoGluon 1.4 forecasting, Mac M4 training, institutional-grade probabilistic forecasts"**

- **Source Prefixing:** ALL columns prefixed with source (`databento_`, `fred_`, `epa_`, etc.)
- **MotherDuck Cloud:** Source-of-truth storage for all data (raw, features, forecasts)
- **Local DuckDB Mirror:** Training landing pad (synced before each training run for fast I/O)
- **AnoFox SQL Macros:** Feature engineering in `database/macros/` (1,428 lines of SQL)
- **AutoGluon 1.4:** Hybrid TabularPredictor + TimeSeriesPredictor with foundation models
- **Mac M4:** ALL training local (TabPFNv2, Mitra, Chronos-Bolt), deterministic, offline-capable

---

## FOR AI ASSISTANTS: CRITICAL CONTEXT

> **This section is essential reading for all AI assistants working on CBI-V15.**

### Current Architecture (V15.1 - AutoGluon Hybrid)

**Data Warehouse:**

- **MotherDuck Cloud**: Source of truth for all data (raw ingestion, features, forecasts)
- **Local DuckDB**: Training landing pad at `data/duckdb/cbi_v15.duckdb` (synced before training)
- **Sync Strategy**: Sync MotherDuck → Local before training for 100-1000x faster I/O

**Forecasting Engine:**

- **AutoGluon 1.4**: Primary engine using hybrid TabularPredictor + TimeSeriesPredictor
  - TabularPredictor with `presets='extreme_quality'` for bucket specialists (includes foundation models)
    - TabularPredictor with `presets='extreme_quality'` for main ZL predictor (all 300+ features)
    - TimeSeriesPredictor (Chronos-Bolt excluded on Mac M4 due to mutex hang - uses alternative models)
    - **Mitra (Salesforce)**: Metal-accelerated time series foundation model via `mitra_trainer.py` (optional fallback)
  - **NO Vertex AI, NO BQML, NO Cloud AutoML**
  - **Mac M4**: All training runs locally on CPU with Metal (MPS) acceleration where available.
  - **Note**: Foundation models (TabPFNv2, Mitra, TabICL) included in `extreme_quality` will run on Mac M4 CPU, but will be significantly slower than with a GPU. The `libomp` fix in `scripts/setup/install_autogluon_mac.sh` is still required for the tree model components (LightGBM, CatBoost, XGBoost).
  - **Mitra Integration**: Available at `src/training/autogluon/mitra_trainer.py` with Metal (MPS) support for Mac M4. See implementation notes in `src/training/README.md`.

**Model Stack** (~90-135 models total):

- **L0**: 9 specialists (8 bucket specialists + 1 main ZL predictor)
  - Each specialist: AutoGluon trains 10-15 models (LightGBM, CatBoost, XGBoost, Neural Nets)
  - Each specialist: AutoGluon creates WeightedEnsemble_L2 (automatic stacking)
  - Output: 9 OOF predictions (P10, P50, P90 quantiles)
- **L1**: Meta-learner (ensemble of 9 specialist ensembles)
  - Input: 9 OOF predictions from L0
  - AutoGluon trains final ensemble (learns optimal specialist weights)
  - Output: Final P10, P50, P90 forecasts for ZL
- **L2**: Production forecasts (uploaded to MotherDuck)
  - Table: `forecasts.zl_predictions` (horizons: 1w, 1m, 3m, 6m)
  - Dashboard queries this table
- **L3**: Monte Carlo simulation (risk metrics ONLY, NOT forecasting)
  - Input: Final L2 predictions (P10, P50, P90)
  - Generates 10,000 scenarios for VaR/CVaR calculation
  - Output: `forecasts.monte_carlo_scenarios` (analytics only)
  - NOT used for trading decisions - just risk reporting

### Architecture Pattern: SQL Features + AutoGluon Training

**Data Collection**: API pull scripts → MotherDuck cloud (scheduled via GitHub Actions)  
**Feature Engineering**: AnoFox SQL macros in `database/macros/` (1,428 lines)  
**Training Prep**: Sync MotherDuck → Local DuckDB (`scripts/sync_motherduck_to_local.py`)  
**Training**: Mac M4 AutoGluon (reads from local DuckDB for speed)  
**Predictions**: Upload to MotherDuck → Dashboard queries `forecasts.zl_predictions`  
**Storage**: MotherDuck cloud (NOT BigQuery)

### Data Sources (V15.1 - Updated Dec 2025)

**Market Data:**

1. **Databento**: 38 futures symbols (ZL/ZS/ZM/CL/HO/RB/HG/6L/DX/etc.) - OHLCV daily

**Economic/Macro:** 2. **FRED**: 24+ macro indicators (rates, yields, VIXCLS, DFEDTARU, DXY, NFCI, STLFSI4) 3. **EIA**: Petroleum products (ULSD wholesale, gasoline, diesel)

**Biofuels (CRITICAL for Biofuel Bucket):** 4. **EPA RIN Prices**: Weekly D3/D4/D5/D6 RIN prices (July 2010-present, FREE)

- URL: https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information
- Weekly volume-weighted averages from EPA EMTS

5. **EIA Biofuels**: Production, consumption, RFS volumes

**Agricultural/Trade:** 6. **USDA FAS Export Sales**: Weekly China soybean purchases (CRITICAL for China bucket) 7. **USDA WASDE**: Monthly supply/demand reports 8. **CFTC COT**: Weekly positioning data (disaggregated reports)

**News/Intelligence (CRITICAL NEW SOURCES):** 9. **Farm Policy News** (MANDATORY): Real-time China/tariff policy

- URL: https://farmpolicynews.illinois.edu/
- Categories: trade, ethanol, budget, regulations, immigration
- Author: Keith Good (University of Illinois)
- **Why Critical**: "China Soybean Buying Deadline", "$11B Farm Aid", Trump policy

10. **farmdoc Daily** (HIGH VALUE): Academic ag economics analysis
    - URL: https://farmdocdaily.illinois.edu/
    - Scott Irwin RIN pricing models (75% R² accuracy)
    - Trade policy analysis, grain outlook, weekly outlook
11. **ScrapeCreators**: Trump posts (Truth Social) + news buckets
12. **ProFarmer**: Weather, basis, barge rates (chris@usoilsolutions.com)

**Weather:** 13. **NOAA**: U.S. weather data 14. **INMET**: Brazil weather stations 15. **Argentina SMN**: Argentina weather observations

**Other:** 16. **Glide API**: Vegas Intel (optional)

### Primary Documents

- `docs/architecture/MASTER_PLAN.md` (this document) – Source of truth for V15.1
- `AGENTS.md` – Agent workspace guide and Big 8 bucket reference
- `database/README.md` – 8-schema layout, SQL macros, feature boundaries
- `.cursorrules` – Cursor-specific rules and conventions

### Critical Rules

1. **NO FAKE DATA** - Only real, verified data from authenticated APIs
2. **ALWAYS CHECK BEFORE CREATING** - Tables, datasets, files, schemas
3. **ALWAYS AUDIT AFTER WORK** - Data quality checks after any data modification
4. **MotherDuck Cloud Storage** - All data in MotherDuck, NOT BigQuery
5. **Local DuckDB Mirror** - MUST sync before training for fast I/O
6. **NO COSTLY RESOURCES** - Approval required for any resource >$5/month
7. **API KEYS** - macOS Keychain (Mac) or `.env` file, never hardcoded
8. **Configuration** - YAML/JSON files, never hardcoded
9. **SQL Macros First** - All features in `database/macros/`, not Python loops
10. **AutoGluon 1.4** - TabularPredictor + TimeSeriesPredictor, NOT manual sklearn
11. **Quantile Regression** - Always `problem_type='quantile'` for P10/P50/P90
12. **Mac Training Only** - All training on Mac M4, no cloud training
13. **ZL Focus** - Soybean Oil Futures (ZL) primary target, Big 8 bucket coverage

### File Organization

- **DuckDB SQL Macros** → `database/macros/` (AnoFox feature engineering)
- **Raw Table Definitions** → `database/models/01_raw/`
- **Python Ingestion** → `src/ingestion/<source>/`
- **AutoGluon Training** → `src/training/autogluon/`
- **Ingestion Scheduling** → `.github/workflows/`
- **Operational Scripts** → `scripts/`
- **Configuration** → `config/`
- **Documentation** → `docs/`

### Workflow (V15.1)

1. **Data Ingestion**: scheduled runs execute `src/ingestion/**` → MotherDuck
2. **Feature Engineering**: AnoFox SQL macros execute → `features.daily_ml_matrix`
3. **Sync to Local**: `python scripts/sync_motherduck_to_local.py` → Local DuckDB
4. **Train Models**: `python src/training/autogluon/train_all_buckets.py` → Reads local DuckDB
5. **Generate Forecasts**: AutoGluon predictions → MotherDuck `forecasts.zl_predictions`
6. **Dashboard**: Vercel Next.js queries MotherDuck directly

### Big 8 Drivers (Complete Coverage Required)

| Bucket            | Features                                      | Primary Data Sources                        |
| ----------------- | --------------------------------------------- | ------------------------------------------- |
| 1. **Crush**      | ZL/ZS/ZM spreads, board crush, oil share      | Databento, NOPA, farmdoc Grain Outlook      |
| 2. **China**      | Export sales, HG-ZS correlation, trade policy | **Farm Policy News: Trade**, USDA FAS       |
| 3. **FX**         | DX, BRL (6L), momentum, volatility            | FRED FX series, Databento                   |
| 4. **Fed**        | Fed funds, curve (T10Y2Y), NFCI, STLFSI4      | FRED, Farm Policy: Budget                   |
| 5. **Tariff**     | Trump sentiment, policy events                | **Farm Policy News: Trade**, ScrapeCreators |
| 6. **Biofuel**    | D4/D5/D6 RIN prices, BOHO spread, biodiesel   | **EPA RIN Prices**, farmdoc RINs, EIA       |
| 7. **Energy**     | CL/HO/RB, crack spreads, CL-ZL correlation    | EIA, Databento                              |
| 8. **Volatility** | VIX, realized vol, stress indices             | FRED VIXCLS, STLFSI4                        |

### Horizons

- 1w (5 trading days)
- 1m (20 trading days)
- 3m (60 trading days)
- 6m (120 trading days)
- 12m (240 trading days) - optional

### Project Structure

- `database/` - DuckDB schema definitions + SQL macros (AnoFox)
- `src/` - Python source code (ingestion, training, engines, models)
  - `src/engines/anofox/` - AnoFox bridge to DuckDB
  - `src/training/autogluon/` - AutoGluon TabularPredictor + TimeSeriesPredictor wrappers
  - (LLM agents removed in V15.1; orchestration now handled directly via AutoGluon + SQL)
  - `src/ingestion/<source>/` - Data collection scripts (per-source)
- `.github/workflows/` - scheduled ingestion workflows
- `data/` - Local DuckDB mirror + model artifacts
- `config/` - Configuration files (YAML/JSON)
- `docs/` - Documentation
- `scripts/` - Operational utilities
- `tests/` - Unit and integration tests
- `dashboard/` - Next.js dashboard (Vercel)

---

## ✅ Locked Features (November 28, 2025)

### Complete Feature Inventory (276 Features)

**Technical Indicators** (19 features):

- Distance MAs: 5 features (EMA 5d, 10d, 21d; SMA 63d, 200d)
- Bollinger: 2 features (%B, Bandwidth)
- PPO: 1 feature (12, 26, 9)
- VWAP: 1 feature (21d distance)
- Volatility: 3 features (Garman-Klass, Parkinson, 21d)
- Microstructure: 2 features (Amihud, OI/Volume)
- Cross-asset: 3 features (BOHO spread, ZL-BRL corr, Terms of Trade)
- Metadata: 2 features (Seasonality SIN/COS)

**FX Indicators** (16 features):

- BRL Momentum: 3 features (21d, 63d, 252d)
- DXY Momentum: 3 features (21d, 63d, 252d)
- BRL Volatility: 2 features (21d, 63d)
- ZL-BRL Correlation: 3 features (30d, 60d, 90d)
- ZL-DXY Correlation: 3 features (30d, 60d, 90d)
- Terms of Trade: 1 feature
- Correlation Regimes: 2 features

**Fundamental Spreads** (5 features):

- Board Crush: `(ZM × 0.022 + ZL × 11) - ZS`
- Oil Share: `(ZL × 11) / Board_Crush_Value`
- Hog Spread: `HE - (0.8 × ZC + 0.2 × ZM)`
- BOHO Spread: `(ZL/100 × 7.5) - HO`
- China Pulse: `CORR(HG, ZS, 60d)`

**Pair Correlations** (112 features):

- 28 pairs × 4 horizons (30d, 60d, 90d, 252d)

**Cross-Asset Betas** (28 features):

- 7 assets × 4 horizons (30d, 60d, 90d, 252d)

**Lagged Features** (96 features):

- 8 symbols × 12 lags (1d, 2d, 3d, 5d, 10d, 21d for prices & returns)

**Total**: **276 features** pre-computed in DuckDB/MotherDuck ✅

### Symbols Locked In (10-12 symbols)

**Commodities** (8 symbols):

- ZL (Soybean Oil) - PRIMARY TARGET
- ZS (Soybeans), ZM (Soybean Meal)
- CL (Crude Oil), HO (Heating Oil)
- FCPO (Palm Oil), ZC (Corn), HE (Lean Hogs)

**FX** (2 symbols):

- 6L (BRL Futures), DX (DXY Futures)

**Optional** (2 symbols):

- HG (Copper) - For China Pulse
- GC (Gold) - For Real-Terms Price

### Prerequisites Before Training

**Must Complete** (Current Status):

1. ⚠️ EPA RIN Prices ingestion (D3/D4/D5/D6 weekly)
2. ⚠️ Farm Policy News scraper (China trade policy - CRITICAL)
3. ⚠️ farmdoc Daily scraper (Scott Irwin RIN analysis)
4. ✅ USDA FAS Export Sales (real API implemented)
5. ⚠️ CFTC COT scheduling (script exists; ensure it is scheduled)
6. ⚠️ Local DuckDB mirror setup (training landing pad)

**Status**: ⚠️ **IN PROGRESS** - Phase 0-1 implementation underway

### MotherDuck Schema Structure

**Schemas**: raw, staging, features, training, forecasts, reference, ops
**Primary Keys**: All tables use `as_of_date` (NOT `date`) for consistency
**Indexing**: DuckDB handles automatically
**Master Feature Table**: `features.daily_ml_matrix` (300+ features)

**Last Updated**: December 9, 2025
