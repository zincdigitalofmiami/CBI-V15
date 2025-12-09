# CBI-V15 System Status - Complete Reference

**Last Updated:** December 7, 2024  
**Status:** Production-Ready Architecture

---

## 📊 System Overview

### Data Pipeline Health

| Component | Status | Count | Location |
|-----------|--------|-------|----------|
| **Data Sources** | ✅ Active | 6 APIs | Databento, FRED, EIA, ScrapeCreators, USDA, CFTC |
| **Ingestion Scripts** | ✅ Ready | 15+ collectors | `src/ingestion/*` |
| **Database Schemas** | ✅ Deployed | 8 schemas | MotherDuck `cbi_v15` |
| **Raw Tables** | ✅ Defined | 15+ tables | `database/definitions/01_raw/` |
| **Feature Tables** | ✅ Defined | 10+ tables | `database/definitions/03_features/` |
| **SQL Macros** | ✅ Loaded | 6 macro files | `database/macros/` |

### Model Stack Health

| Level | Component | Status | Files |
|-------|-----------|--------|-------|
| **L1** | Base Models | ✅ 3 families | lightgbm, catboost, xgboost |
| **L2** | Meta-Learner | ✅ Implemented | `model_sweep.py` |
| **L3** | QRA Ensemble | ✅ Implemented | `src/ensemble/qra_ensemble.py` |
| **L4** | Monte Carlo | ✅ Implemented | `src/simulators/monte_carlo_sim.py` |

### TSci Agent Health

| Agent | OpenAI? | Guardrails? | Output | Status |
|-------|---------|-------------|--------|--------|
| Curator | ✅ Yes | ✅ Yes | JSON (qa_checks) | ✅ Ready |
| Planner | ✅ Yes | ✅ Yes | JSON (jobs config) | ✅ Ready |
| Forecaster | ✅ Yes | ✅ Yes | QRA weights + pipeline | ✅ Ready |
| Reporter | ✅ Yes | ✅ Yes | HTML + JSON reports | ✅ Ready |

---

## 🗄️ Database Architecture

### MotherDuck Production

**Database:** `cbi_v15` (underscore, not hyphen)  
**Connection:** `md:cbi_v15?motherduck_token={token}`  
**Console:** https://app.motherduck.com/

### Schemas (8 total)

```sql
raw         -- Raw ingestion (databento, fred, eia, scrapecreators, cftc, usda, weather)
staging     -- Cleaned/normalized (market, crush, china, news)
features    -- Engineered features (300+ features, Big 8 buckets, daily_ml_matrix)
training    -- Training datasets (with targets, splits, regime tags)
forecasts   -- Model predictions (zl_predictions with horizon column)
reference   -- Calendars, catalogs, metadata
ops         -- Pipeline metrics, logs, ingestion status
tsci        -- TSci jobs, runs, qa_checks, simulations
```

### Tables (30+ total)

**Raw Layer (15+ tables):**
- `raw.databento_ohlcv_daily` (38 futures symbols)
- `raw.fred_macro` (24 macro series)
- `raw.eia_biofuels` (RIN prices, biodiesel)
- `raw.scrapecreators_trump`, `raw.scrapecreators_news_buckets`
- `raw.cftc_cot` (COT data for all symbols)
- `raw.usda_wasde`, `raw.usda_export_sales`, `raw.usda_crop_progress`
- `raw.weather_station_daily`, `raw.weather_forecast_grid`

**Features Layer (10+ tables):**
- `features.technical_indicators_all_symbols` (93 indicators × 38 symbols)
- `features.cross_asset_correlations`
- `features.fundamental_spreads`
- `features.big8_bucket_scores` (Crush, China, FX, Fed, Tariff, Biofuel, Energy, **Volatility**)
- `features.daily_ml_matrix_zl` ← **MASTER TABLE (300+ features)**

**Training Layer:**
- `training.daily_ml_matrix_zl` (with train/val/test splits)

**TSci Layer:**
- `tsci.jobs` (planned experiments)
- `tsci.runs` (model sweep results)
- `tsci.qa_checks` (data quality)
- `tsci.simulations` (Monte Carlo results)

---

## 🤖 Models & Algorithms

### Base Models (L1)

| Model Family | Implementation | Horizons | Quantiles | Status |
|--------------|----------------|----------|-----------|--------|
| **LightGBM** | `src/training/baselines/lightgbm_zl.py` | 1w, 1m, 3m, 6m, 12m | P50 | ✅ Implemented |
| **CatBoost** | `src/training/baselines/catboost_zl.py` | 1w, 1m, 3m, 6m, 12m | P10/P50/P90 | ✅ Implemented |
| **XGBoost** | `src/training/baselines/xgboost_zl.py` | 1w, 1m, 3m, 6m, 12m | P10/P50/P90 | ✅ Implemented |
| **TFT** | Planned | All | P10/P50/P90 | ⚠️ Future |
| **Prophet** | Planned | All | Intervals | ⚠️ Future |
| **GARCH** | Planned | Short | Volatility | ⚠️ Future |

### Meta-Learning (L2)

**Module:** `src/models/tsci/model_sweep.py`

**Features:**
- Trains multiple candidate models per bucket/horizon
- Evaluates on validation set (RMSE, pinball loss, coverage)
- Selects best model per sweep
- Logs results to `tsci.runs`

**Big 8 Role:** Focus tags for reporting, NOT feature filters. Models see all 300+ features.

### Ensemble (L3)

**Module:** `src/ensemble/qra_ensemble.py`

**Algorithm:** Quantile Regression Averaging (QRA)
- Weighted combination of model quantiles
- Regime-aware weights (TSci suggests, QRA executes)
- Preserves full uncertainty structure

**Inputs:** Quantile forecasts from L1/L2  
**Outputs:** Combined P10/P50/P90 distributions

### Risk Simulation (L4)

**Module:** `src/simulators/monte_carlo_sim.py`

**Algorithm:** Monte Carlo path simulation
- Generates 1,000 probabilistic price paths
- Estimates from QRA quantiles (P10/P50/P90)
- Calculates downside risk metrics

**Outputs:**
- VaR (5%): 5th percentile of final prices
- CVaR (5%): Mean of worst 5% outcomes
- Max Drawdown: Worst decline from initial price
- Scenario statistics for dashboard

---

## 🧠 TSci Agents (OpenAI-Powered)

### CuratorAgent

**Purpose:** Data quality & hygiene  
**LLM Task:** Classify data quality, suggest QA actions  
**System Prompt:** "Never invent data, only reason over provided metrics"  
**Output:** JSON for `tsci.qa_checks`

**Example:**
```python
from src.models.tsci.curator import CuratorAgent

agent = CuratorAgent()
result = agent.analyze_data_quality("raw.databento_ohlcv_daily")
# Returns: {"data_quality": "pass", "outlier_strategy": "clip", ...}
```

### PlannerAgent

**Purpose:** Model selection strategy  
**LLM Task:** Suggest model candidates and hyperparameter bands  
**System Prompt:** "Big 8 are overlays; models see all features"  
**Output:** Job configs for `tsci.jobs`

**Example:**
```python
from src.models.tsci.planner import suggest_model_candidates

result = suggest_model_candidates(
    bucket="volatility",  # Note: spelled out, not "vol"
    horizon="1w",
    regime="high_volatility",
)
# Returns: {"candidate_models": ["catboost", "tft"], "hyperparam_ranges": {...}}
```

### ForecasterAgent

**Purpose:** Ensemble orchestration  
**LLM Task:** Recommend QRA weights based on regime  
**Execution:** Runs QRA (L3) → Monte Carlo (L4) numerically  
**Output:** Forecast distributions + risk metrics

**Example:**
```python
from src.models.tsci.forecaster import ForecasterAgent

agent = ForecasterAgent()
result = agent.run_full_forecast_pipeline(
    quantile_forecasts=[...],  # From trained models
    weights={"lightgbm": 0.4, "catboost": 0.6},
    regime="high_volatility",
)
# Returns: ensemble forecast + Monte Carlo risk metrics
```

### ReporterAgent

**Purpose:** Narrative generation  
**LLM Task:** Create web-ready reports  
**System Prompt:** "Volatility = VIX/vol regimes, NOT trading volume"  
**Output:** HTML + JSON for `/quant-admin`

**Example:**
```python
from src.models.tsci.reporter import ReporterAgent

agent = ReporterAgent()
report = agent.generate_report(
    run_id="2024-12-07T12:00:00Z",
    forecast_data={...},
    bucket_contributions={"crush": 0.42, "volatility": 0.27},
)
# Returns: {summary_html, drivers, scenarios, confidence}
```

---

## 🔑 API Keys & Environment

### Required Keys

```bash
# Database
export MOTHERDUCK_TOKEN="your_token"
export MOTHERDUCK_DB="cbi_v15"

# AI/LLM
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-5.1"  # Or custom: ft:gpt-4.1:org:zl-tsci

# Market Data
export DATABENTO_API_KEY="your_key"

# Economic Data
export FRED_API_KEY="your_key"
export EIA_API_KEY="your_key"

# News/Sentiment
export SCRAPECREATORS_API_KEY="your_key"
```

### Storage Locations

1. **Local Development:** `.env` file (gitignored)
2. **macOS Scripts:** Keychain (via `store_api_keys.sh`)
3. **Dashboard (Vercel):** Environment variables (via Terraform or Vercel UI)
4. **GCP (Legacy):** Secret Manager (optional, not used in V15)

---

## 📦 Data Coverage

### Market Data (Databento)

**Symbols:** 38 total
- Agricultural: ZL, ZS, ZM, ZC, ZW, ZO, ZR, HE, LE, GF, FCPO (11)
- Energy: CL, HO, RB, NG (4)
- Metals: HG, GC, SI, PL, PA (5)
- FX Futures: 6E, 6J, 6B, 6C, 6A, 6N, 6M, 6L, 6S, DX (10)
- Treasuries: ZF, ZN, ZB (3)
- FX Spot: 13 pairs (derived)

### Economic Data (FRED)

**Series:** 24 total
- Rates: FEDFUNDS, DGS1MO–DGS30 (11 series)
- Spreads: T10Y2Y, T10Y3M, TEDRATE (3 series)
- Financial Conditions: NFCI, STLFSI4 (2 series)
- Economic: UNRATE, CPIAUCSL, GDP, PAYEMS (4 series)
- Market: VIXCLS, DTWEXBGS, DTWEXAFEGS, DTWEXEMEGS (4 series)

### Biofuels Data (EIA)

**Series:** 6 total
- RIN_D4, RIN_D6 (prices)
- BIODIESEL_PROD, BIODIESEL_CONSUMPTION (volume)
- RFS_VOLUMES, ULSD_WHOLESALE_MIDWEST

### News & Sentiment (ScrapeCreators)

**Buckets:** 8 total (4 active, 4 planned)
- Active: biofuel_policy, china_demand, tariffs_trade, trump_truth_social
- Planned: market_volatility, crop_failures, supply_chain, general_market

### Additional Sources

- **CFTC:** COT data for all 38 futures
- **USDA:** WASDE, export sales, crop progress
- **Weather:** NOAA (14 regions: Brazil, Argentina, US)

---

## 🔧 Utilities & Tools

### Python Utilities (`src/utils/`)

- `openai_client.py` - OpenAI integration with model switching
- `keychain_manager.py` - macOS Keychain for secure credential storage

### Scripts (`scripts/`)

**Setup:**
- `setup_database.py` - Deploy schemas to MotherDuck/DuckDB
- `setup/store_api_keys.sh` - Store API keys in Keychain
- `setup/test_motherduck_connection.py` - Verify MotherDuck connection

**Ops:**
- `ops/test_connections.py` - Test all API connections
- `ops/check_data_availability.py` - Check data freshness
- `ops/ingestion_status.py` - Ingestion pipeline status
- `ops/vegas/collect_vegas_events.py` - Vegas events collection

**Export:**
- `export/databento_to_motherduck.py` - Export market data

---

## 🎯 Big 8 Buckets (Focus Overlays)

| # | Bucket ID | Official Name | Key Features |
|---|-----------|---------------|--------------|
| 1 | `crush` | Crush | ZL/ZS/ZM spreads, oil share, board crush |
| 2 | `china` | China | HG copper, export sales, China weather |
| 3 | `fx` | FX | DX, BRL, CNY, MXN, dollar momentum |
| 4 | `fed` | Fed | Fed funds, yield curve, NFCI |
| 5 | `tariff` | Tariff | Trump sentiment, Section 301, trade policy |
| 6 | `biofuel` | Biofuel | RIN prices, biodiesel, RFS, BOHO spread |
| 7 | `energy` | Energy | Crude, HO, RB, crack spreads |
| 8 | `volatility` | **Volatility** | **VIX, realized vol, STLFSI4, stress indices** |

**⚠️ CRITICAL:** Bucket 8 is **Volatility** (price variance), NOT volume (trading activity).

**Usage:**
- Models see ALL 300+ features from Anofox
- Big 8 used for tagging, regime weighting, driver attribution
- NOT hard filters or exclusive feature sets

---

## 📚 Documentation Index

### Core Architecture
- `docs/project_docs/tsci_anofox_architecture.md` - TSci ↔ Anofox integration
- `docs/architecture/META_LEARNING_FRAMEWORK.md` - AutoML design
- `docs/architecture/ENSEMBLE_ARCHITECTURE_PROPOSAL.md` - L1–L4 stack

### Operations
- `docs/ops/NAMING_CONVENTIONS.md` - Volatility vs volume rules (MANDATORY)
- `docs/ops/BIG_8_BUCKETS_REFERENCE.md` - Big 8 as focus, not cages
- `docs/ops/OPENAI_CUSTOM_MODEL_GUIDE.md` - Fine-tuning TSci
- `docs/ops/STRUCTURE_ALIGNMENT_PLAN.md` - Dataform-style organization

### Data Sources
- `DATA_LINKS_MASTER.md` - All API endpoints and credentials
- `WEB_SCRAPING_TARGETS_MASTER.md` - 59 web sources by tier
- `docs/data_sources/DATA_SOURCES_MASTER.md` - Complete data inventory

### Setup & Deployment
- `DATABASE_SETUP_GUIDE.md` - MotherDuck/DuckDB setup
- `docs/setup/API_KEYS_SETUP.md` - Credential management
- `SECURITY_AND_CONFIG_AUDIT.md` - Security audit results

### Status & Completion
- `TSCI_ANOFOX_UPGRADE_COMPLETE.md` - Latest upgrade (Dec 7, 2024)
- `PHASE_A_COMPLETE.md` - Structural cleanup
- `FEATURE_ENGINEERING_COMPLETE.md` - Feature work
- `CFTC_COT_INGESTION_COMPLETE.md` - CFTC integration

---

## 🚀 Quick Start Commands

### 1. Setup Database
```bash
export MOTHERDUCK_TOKEN="your_token"
export MOTHERDUCK_DB="cbi_v15"
python scripts/setup_database.py --both
```

### 2. Run Ingestion
```bash
python src/ingestion/databento/collect_daily.py
python src/ingestion/fred/collect_fred_fx.py
python src/ingestion/eia/collect_eia_biofuels.py
```

### 3. Build Features
```bash
python src/engines/anofox/build_features.py
python src/engines/anofox/build_training.py
```

### 4. Train Models
```bash
# Individual baselines
python src/training/baselines/lightgbm_zl.py
python src/training/baselines/catboost_zl.py

# Or TSci-orchestrated sweep
python src/models/tsci/planner.py
```

### 5. Generate Forecasts
```python
from src.models.tsci.forecaster import ForecasterAgent

agent = ForecasterAgent()
result = agent.run_full_forecast_pipeline(
    quantile_forecasts=forecasts,  # From trained models
    weights={"lightgbm": 0.4, "catboost": 0.6},
    regime="high_volatility",
)
```

### 6. Start Dashboard
```bash
cd dashboard
npm install
npm run dev
```

---

## 📦 Dependencies

### Python (requirements.txt)
```
pandas>=2.2.3
numpy>=1.26.4
duckdb>=1.1.3
databento>=0.40.0
lightgbm>=4.5.0
catboost>=1.2.7
xgboost>=2.1.1
scikit-learn>=1.5.2
openai>=1.30.0
python-dotenv>=1.0.1
pyyaml>=6.0.2
```

### Node.js (dashboard/package.json)
```
next>=14.0.0
@motherduck/wasm-client
react, recharts, nivo
```

---

## ⚙️ Configuration

### Environment Variables
See: `config/env-templates/env.template`

### Data Source Configs
- `config/ingestion/sources.yaml` - API endpoints, rate limits, schedules
- `config/training/model_config.yaml` - Model hyperparameters, horizons
- `config/data_sources.yaml` - Complete data source registry

---

## 🔐 Security

### API Keys Storage
1. **Development:** `.env` file (gitignored)
2. **macOS:** Keychain (`scripts/setup/store_api_keys.sh`)
3. **Production:** Vercel env vars or GCP Secret Manager

### Secrets Management
- ✅ `.env`, `web/.env.local` in `.gitignore`
- ✅ `_archive/` contains old exposed secrets (gitignored)
- ⚠️ **TODO:** Rotate MotherDuck + OpenAI tokens if exposed

---

## 📊 Current Metrics

### Feature Coverage
- **Total Features:** 300+ engineered features
- **Symbols Covered:** 38 futures + 13 FX spot
- **Big 8 Features:** ~80 bucket-specific features
- **Technical Indicators:** 93 per symbol (RSI, MACD, BB, ATR, etc.)
- **Macro Series:** 24 from FRED

### Model Performance (Baseline)
- **LightGBM:** R² ~0.95, MAE ~0.30 (on test set)
- **CatBoost:** Pinball loss ~0.25 (P50)
- **XGBoost:** Coverage ~88% (90% target for P10/P90)

---

## 🎯 Next Steps (Production)

1. **Load Historical Data**
   - Backfill Databento (2020–present)
   - Backfill FRED, EIA, ScrapeCreators
   
2. **Train Production Models**
   - Run all 3 baselines on real data
   - Execute TSci model sweeps
   - Save winning models to `models/`

3. **Deploy QRA + Monte Carlo**
   - Generate production forecasts
   - Calculate risk metrics
   - Log to `forecasts.*` tables

4. **Wire to Dashboard**
   - Create `/api/forecasts` route (MotherDuck)
   - Create `/api/tsci/reports` route
   - Display in `/quant-admin`

5. **Fine-Tune TSci**
   - Collect example decisions
   - Fine-tune OpenAI model
   - Optionally add RFT

---

## ✅ System Health Checklist

### Daily
- [ ] Check `scripts/ops/ingestion_status.py` for data freshness
- [ ] Review `/quant-admin` for TSci reports
- [ ] Monitor MotherDuck table sizes

### Weekly
- [ ] Run model sweeps for each horizon
- [ ] Compare ensemble performance to baselines
- [ ] Review Big 8 bucket contributions

### Monthly
- [ ] Full model retraining
- [ ] Update regime weights
- [ ] Audit data quality (`tsci.qa_checks`)

---

**For detailed guides, see:**
- Setup: `DATABASE_SETUP_GUIDE.md`
- Training: `src/training/baselines/README.md`
- TSci: `src/models/tsci/README.md`
- APIs: `DATA_LINKS_MASTER.md`




