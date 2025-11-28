# MASTER PLAN
**Date:** November 28, 2025  
**Last Revised:** November 28, 2025  
**Status:** V15 Clean Architecture  
**Purpose:** Single source of truth for CBI-V15 architecture

---

## PHILOSOPHY

**"Clean architecture, Dataform-first ETL, Mac M4 training, institutional-grade"**

- **Source Prefixing:** ALL columns prefixed with source (`databento_`, `fred_`, etc.)
- **Mac M4:** ALL training + ALL feature engineering (100% local, deterministic)
- **BigQuery:** Source-of-truth storage + Dataform ETL + Dashboard (NOT training)
- **Dataform:** Primary ETL framework (version controlled, CI/CD integrated)
- **External Drive:** Local cache + backup (never authoritative; reproducible from BigQuery)

---

## FOR AI ASSISTANTS: CRITICAL CONTEXT

> **This section is essential reading for all AI assistants working on CBI-V15.**

### Current Architecture (V15)
- Apple M4 Mac handles every training and inference task (PyTorch MPS + LightGBM + TensorFlow Metal)
- BigQuery = storage + Dataform ETL + dashboard read layer only
- NO BigQuery ML, NO AutoML, NO Vertex AI training
- Predictions generated locally, uploaded with `scripts/upload/upload_predictions.py`
- Dataform handles all ETL transformations (raw → staging → features → training)

### Architecture Pattern: Dataform ETL + Mac Training
**Data Collection**: Python scripts → BigQuery (via Dataform declarations)  
**ETL**: Dataform (BigQuery SQL transformations)  
**Feature Engineering**: 
  - Dataform SQL: Correlations, moving averages, regimes, Big 8 drivers
  - Python: Complex sentiment (FinBERT), policy extraction, technical indicators (pandas-ta)
**Training**: Mac M4 (LightGBM, PyTorch/TFT)  
**Storage**: BigQuery (us-central1 only)

### Data Sources (V15)
1. **Databento**: Market data (GLBX.MDP3) - futures, FX, options (2000→present)
2. **FRED**: 55-60 economic series (rates, inflation, employment, GDP, DXY, VIX, PPOILUSDM)
3. **NOAA**: Weather data (CDO, GFS forecasts)
4. **INMET**: Brazil weather stations
5. **Argentina SMN**: Argentina weather observations
6. **Google Public Datasets**: GSOD, GFS, Normals, GDELT, BLS, FEC, COVID
7. **USDA**: NASS reports (WASDE, crop progress, exports), FAS Export Sales Reports
8. **CFTC**: Commitments of Traders (weekly)
9. **EIA**: Biofuels (production, RIN prices, RFS volumes)
10. **ScrapeCreators**: Trump posts (Truth Social) + news buckets (biofuel, China, tariffs)
11. **Glide API**: Vegas Intel (restaurants, casinos, shifts, events)
12. **World Bank**: Optional alternative macro series (documented use only)

### Primary Documents
- `docs/architecture/MASTER_PLAN.md` (this document) – Source of truth
- `docs/architecture/DATAFORM_ARCHITECTURE.md` – Dataform structure guide
- `docs/plans/TRAINING_PLAN.md` – Training strategy
- `docs/reference/AI_ASSISTANT_GUIDE.md` – AI assistant quick start

### Critical Rules
1. **NO FAKE DATA** - Only real, verified data from authenticated APIs
2. **ALWAYS CHECK BEFORE CREATING** - Tables, datasets, files, schemas
3. **ALWAYS AUDIT AFTER WORK** - Data quality checks after any data modification
4. **us-central1 ONLY** - All BigQuery datasets, GCS buckets, GCP resources
5. **NO COSTLY RESOURCES** - Approval required for any resource >$5/month
6. **API KEYS** - macOS Keychain (Mac) or Secret Manager (GCP scheduler)
7. **Configuration** - YAML/JSON files, never hardcoded
8. **Dataform First** - All ETL transformations in Dataform, version controlled
9. **Mac Training Only** - All training on Mac M4, no cloud training
10. **ZL Focus** - Soybean Oil Futures (ZL) primary target, single-asset multi-horizon

### File Organization
- **Dataform SQL** → `dataform/definitions/`
- **Python scripts** → `src/`
- **Operational scripts** → `scripts/`
- **Configuration** → `config/`
- **Documentation** → `docs/`

### Workflow
1. Run data quality checks: `python scripts/validation/data_quality_checks.py`
2. Export training data: `python scripts/export/export_training_data.py --horizon 1m`
3. Train models: `python src/training/baselines/lightgbm_zl.py --horizon 1m`
4. Generate predictions: `python src/prediction/generate_forecasts.py`
5. Upload predictions: `python scripts/upload/upload_predictions.py`
6. Run Dataform: `cd dataform && dataform run`

### Big 8 Drivers (Complete Coverage Required)
1. Crush Margin (ZS + ZM - ZL)
2. China Imports (USDA exports + sentiment)
3. Dollar Index (FRED DTWEXBGS)
4. Fed Policy (Fed funds, yield curve)
5. Tariffs (Policy events, intensity)
6. Biofuels (Biodiesel margin + RIN prices)
7. Crude Oil (CL price, ZL-CL correlation)
8. VIX (VIX level, regime classification)

### Horizons
- 1w (5 trading days)
- 1m (20 trading days)
- 3m (60 trading days)
- 6m (120 trading days)
- 12m (240 trading days) - optional

### Project Structure
- `dataform/` - All BigQuery ETL (Dataform)
- `src/` - All Python source code
- `config/` - Configuration files (YAML/JSON)
- `docs/` - Documentation
- `scripts/` - Operational utilities
- `tests/` - Unit and integration tests
- `dashboard/` - Next.js dashboard (Vercel)

---

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

**Total**: **276 features** pre-computed in BigQuery ✅

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

### Prerequisites Before Baselines

**Must Complete**:
1. ✅ USDA Ingestion (WASDE, crop progress, exports)
2. ✅ CFTC Ingestion (COT positions, managed money)
3. ✅ EIA Ingestion (RIN prices, biodiesel production)

**Status**: ⚠️ **REQUIRED** before baseline training

### BigQuery Structure

**Partitioning**: All tables `PARTITION BY DATE(date)`
**Clustering**: `CLUSTER BY symbol` (where applicable)
**No Joins**: Skeleton structure = table definitions only
**Master Join**: `daily_ml_matrix` joins all features

**Last Updated**: November 28, 2025

