# CBI-V15 Intelligence Platform

> **Read first – fast-moving workspace.** Always check the latest work before editing: `docs/architecture/MASTER_PLAN.md`, `AGENTS.md`, `DATA_LINKS_MASTER.md`, and the active master plan `.cursor/plans/ALL_PHASES_INDEX.md`. Multiple agents work in parallel—plans can drift; verify current files, avoid duplicating scripts/MDs/folders, and keep the explorer clean.

> **ZL (Soybean Oil) forecasting system** — DuckDB/MotherDuck + AutoGluon 1.4 + Trigger.dev, with SQL-first features.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MotherDuck](https://img.shields.io/badge/Database-MotherDuck-blue)](https://motherduck.com)
[![Next.js](https://img.shields.io/badge/Dashboard-Next.js%2014-black)](https://nextjs.org)

---

## 🎯 Overview

**CBI-V15** is a production-grade commodity forecasting intelligence platform for **ZL (Soybean Oil) futures**, combining institutional quantitative methods with modern ML infrastructure.

### Core Components

- **SQL-First Feature Engineering** — 1,428 lines of AnoFox SQL macros in `database/macros/` generating 300+ features across 8 economic buckets (Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility)
- **AutoGluon 1.4 Hybrid Engine** — Multi-layer ensemble architecture:
  - **L0**: 9 specialist models (8 bucket specialists + 1 main ZL predictor) using TabularPredictor with `extreme_quality` presets
  - **L1**: Meta-learner ensemble fusing specialist outputs
  - **L2**: Production forecasts (P10/P50/P90 quantiles) at 1w/1m/3m/6m horizons
  - **L3**: Monte Carlo simulation for VaR/CVaR risk metrics (analytics only)
- **Mac M4 Training** — All training runs locally on CPU with Metal (MPS) acceleration where available; includes foundation models (TabPFNv2, Mitra, TabICL)
- **Trigger.dev Orchestration** — Automated ingestion from 10+ data sources (Databento, FRED, EIA/EPA RINs, USDA FAS/WASDE, CFTC COT, Farm Policy News, farmdoc Daily)
- **MotherDuck + Local DuckDB** — Cloud-first data warehouse (MotherDuck = source of truth; local mirror synced before training for 100-1000x faster I/O)
- **Next.js 14 Dashboard** — Real-time ZL price charts, multi-horizon forecast fans, confidence metrics, Big 8 health panel; queries MotherDuck directly via DuckDB-WASM

---

## 🏗️ Architecture (V15.1)

### Data Flow

1. **Ingestion Layer** (`trigger/`)
   - Trigger.dev jobs write to MotherDuck cloud warehouse (`raw.*` schema)
   - Organized per source: `DataBento`, `FRED`, `EIA_EPA`, `USDA`, `CFTC`, `ScrapeCreators`, `ProFarmer`, `UofI_Feeds`, `Weather`, `Vegas`, `TradingEconomics`, `Adapters`
   - All columns prefixed with source (`databento_`, `fred_`, `epa_`, etc.)

2. **Feature Engineering** (`database/macros/`)
   - AnoFox SQL macros transform raw data → `features.*` schema
   - 1,428 lines of SQL generating 300+ features
   - Bucket-specific selectors in `config/bucket_feature_selectors.yaml`

3. **Training Preparation**
   - Sync MotherDuck → local DuckDB via `scripts/sync_motherduck_to_local.py`
   - Local mirror at `data/duckdb/cbi_v15.duckdb` (100-1000x faster I/O)

4. **Model Training** (`src/training/autogluon/`)
   - **L0 Specialists**: 9 AutoGluon TabularPredictors (8 buckets + 1 main ZL)
     - Each trains 10-15 models (LightGBM, CatBoost, XGBoost, NNs, foundation models)
     - Each creates WeightedEnsemble_L2 (automatic stacking)
   - **L1 Meta-Learner**: Ensemble of 9 specialist ensembles
   - Training artifacts saved to `artifacts/models/` (excluded from git)

5. **Forecast Production**
   - Upload predictions to MotherDuck (`forecasts.zl_predictions`)
   - Multi-horizon outputs: 1w, 1m, 3m, 6m
   - Quantile forecasts: P10, P50, P90

6. **Risk Analytics** (`src/simulators/`)
   - Monte Carlo simulation in `monte_carlo_sim.py`
   - Generates 10,000 scenarios for VaR/CVaR calculation
   - Output: `forecasts.monte_carlo_scenarios` (analytics only, NOT trading signals)

7. **Dashboard** (`dashboard/`)
   - Next.js 14 app queries MotherDuck directly via DuckDB-WASM
   - Real-time ZL price charts (Databento API, 5min updates)
   - Multi-horizon forecast fans with confidence bands
   - Big 8 bucket health panel

---

## 📁 Project Structure

```
CBI-V15/
├── trigger/                      # Trigger.dev ingestion orchestration
│   ├── DataBento/                # Futures OHLCV (38 symbols)
│   ├── FRED/                     # Macro indicators (24+)
│   ├── EIA_EPA/                  # Energy & biofuel data (EPA RIN prices)
│   ├── USDA/                     # Export sales, WASDE reports
│   ├── CFTC/                     # COT positioning data
│   ├── ScrapeCreators/           # Farm Policy News, farmdoc Daily
│   ├── ProFarmer/                # Pro Farmer intelligence
│   ├── UofI_Feeds/               # University of Illinois feeds
│   ├── Weather/                  # NOAA, INMET, SMN weather data
│   ├── Vegas/                    # Las Vegas demand intelligence (Glide API)
│   ├── TradingEconomics/         # Global economics data
│   ├── Adapters/                 # Shared ingestion utilities
│   ├── DirectScrapers/           # Custom web scrapers
│   └── Orchestration/            # Job scheduling & coordination
├── database/
│   ├── ddl/                      # 54 DDL files (V15.1 schema)
│   ├── macros/                   # 1,428 lines of AnoFox SQL macros
│   ├── seeds/                    # Reference data
│   ├── migrations/               # Schema version control
│   └── tests/                    # SQL test fixtures
├── src/
│   ├── training/
│   │   ├── autogluon/            # AutoGluon 1.4 hybrid engine
│   │   │   ├── mitra_trainer.py  # Salesforce Mitra (Metal-accelerated TS)
│   │   │   └── timeseries_trainer.py  # TimeSeriesPredictor wrapper
│   │   ├── baselines/            # Baseline models (ARIMA, ETS, etc.)
│   │   └── utils/                # Training utilities
│   ├── simulators/
│   │   └── monte_carlo_sim.py    # VaR/CVaR risk analytics
│   ├── reporting/
│   │   └── training_auditor.py   # Training run audits
│   ├── features/                 # Feature engineering utilities
│   ├── engines/anofox/           # AnoFox SQL macro bridge
│   ├── ingestion/                # Legacy ingestion (migrated to trigger/)
│   ├── ensemble/                 # Ensemble utilities
│   ├── shared/                   # Shared utilities
│   ├── data/                     # Data loading utilities
│   └── utils/                    # General utilities
├── dashboard/
│   ├── app/
│   │   ├── api/
│   │   │   ├── big8/             # Big 8 bucket API routes
│   │   │   ├── forecasts/        # Forecast API routes (ZL)
│   │   │   ├── health/           # Data source health checks
│   │   │   ├── models/           # Model metrics API
│   │   │   ├── live/             # Live price data (Databento)
│   │   │   ├── training/         # Training status API
│   │   │   └── shap/             # SHAP explainability API
│   │   └── page.tsx              # Main dashboard (ZL chart + forecasts)
│   ├── components/
│   │   ├── big8/                 # Big8Panel component
│   │   ├── charts/               # ForecastFanChart, price charts
│   │   └── metrics/              # ConfidenceBadge, metric displays
│   └── lib/                      # Dashboard utilities
├── scripts/
│   ├── setup/                    # Environment setup (install_autogluon_mac.sh)
│   ├── sync_motherduck_to_local.py  # Cloud → local sync
│   ├── validation/               # Data validation scripts
│   └── test_*.py                 # Integration tests
├── config/
│   ├── requirements/             # Python dependencies
│   ├── bucket_feature_selectors.yaml  # Big 8 bucket feature configs
│   ├── data_sources.yaml         # Data source configurations
│   ├── training/                 # Training configs
│   ├── ingestion/                # Ingestion configs
│   ├── env-templates/            # .env.example templates
│   └── schedulers/               # Cron/scheduling configs
├── docs/
│   ├── architecture/             # MASTER_PLAN.md, design docs
│   ├── ops/                      # Operations guides
│   └── api/                      # API documentation
├── data/
│   └── duckdb/                   # Local DuckDB mirror (cbi_v15.duckdb)
├── archive/                      # Archived legacy code
├── Justfile                      # Just task runner recipes
├── qodana.yaml                   # Code quality config
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (Mac M4 recommended for optimal AutoGluon performance)
- **Node.js 18+** (for dashboard)
- **Just** task runner (`brew install just`)
- **MotherDuck account** with access token
- **API keys**: Databento, FRED (minimum); optional: EIA, USDA NASS, ScrapeCreator

### Initial Setup

1. **Clone and bootstrap environment**

   ```bash
   git clone https://github.com/zincdigitalofmiami/CBI-V15.git
   cd CBI-V15
   just setup  # Creates .venv, installs Python deps, runs npm ci
   ```

2. **Configure environment variables**

   ```bash
   cp config/env-templates/.env.example .env
   # Edit .env with your credentials:
   # - MOTHERDUCK_DB=cbi_v15
   # - MOTHERDUCK_TOKEN=<your_token>
   # - DATABENTO_API_KEY=<your_key>
   # - FRED_API_KEY=<your_key>
   ```

3. **Install AutoGluon with Mac M4 optimizations** (Mac only)
   ```bash
   source .venv/bin/activate
   scripts/setup/install_autogluon_mac.sh
   ```

### Development Workflows

**Run dashboard locally** (queries MotherDuck cloud)

```bash
just dev  # Starts Trigger.dev (port 3000) + Dashboard (port 3001)
```

**Sync MotherDuck → local DuckDB** (before training)

```bash
source .venv/bin/activate
python scripts/sync_motherduck_to_local.py --dry-run  # Preview changes
python scripts/sync_motherduck_to_local.py            # Execute sync
```

**Run quality checks**

```bash
just qa  # Runs ruff linter
```

**Database operations**

```bash
just db:ddl     # Deploy DDL to MotherDuck
just db:macros  # Deploy SQL macros to MotherDuck
just db:seed    # Load seed data
```

**Autosave (backup commits every 5min)**

```bash
just autosave  # Ctrl+C to stop
```

---

## 🔑 Environment Variables

### Required

- `MOTHERDUCK_DB` — MotherDuck database name (default: `cbi_v15`)
- `MOTHERDUCK_TOKEN` — MotherDuck service token
- `DATABENTO_API_KEY` — Databento API key (38 futures symbols)
- `FRED_API_KEY` — FRED API key (24+ macro indicators)

### Optional (ingestion enhancers)

- `EIA_API_KEY` — Energy Information Administration
- `USDA_NASS_API_KEY` — USDA National Agricultural Statistics Service
- `SCRAPECREATOR_API_KEY` — ScrapeCreator API (news scraping)
- `TRADINGECONOMICS_API_KEY` — TradingEconomics API
- `GLIDE_API_KEY` — Glide API (Vegas demand intelligence)

**Security**: Store secrets in `.env` (gitignored) or macOS Keychain. Never commit credentials.

---

## 📚 Key Documentation

### Architecture & Design

- **[docs/architecture/MASTER_PLAN.md](docs/architecture/MASTER_PLAN.md)** — Single source of truth for V15.1 AutoGluon hybrid architecture
- **[docs/architecture/FEATURE_ENGINEERING_ARCHITECTURE.md](docs/architecture/FEATURE_ENGINEERING_ARCHITECTURE.md)** — AnoFox SQL macro design patterns
- **[docs/architecture/META_LEARNING_FRAMEWORK.md](docs/architecture/META_LEARNING_FRAMEWORK.md)** — Multi-layer ensemble strategy
- **[docs/architecture/ENSEMBLE_ARCHITECTURE_PROPOSAL.md](docs/architecture/ENSEMBLE_ARCHITECTURE_PROPOSAL.md)** — L0/L1/L2 ensemble design
- **[docs/architecture/BASELINE_V15_STRATEGY.md](docs/architecture/BASELINE_V15_STRATEGY.md)** — Baseline model strategy

### Operations & Deployment

- **[docs/ops/MOTHERDUCK_VERCEL_CONNECTION_AUDIT.md](docs/ops/MOTHERDUCK_VERCEL_CONNECTION_AUDIT.md)** — Dashboard connection architecture
- **[docs/ops/deployment_checklist.md](docs/ops/deployment_checklist.md)** — Production deployment guide
- **[docs/ops/POST_REFACTOR_HARDENING_REPORT.md](docs/ops/POST_REFACTOR_HARDENING_REPORT.md)** — V15.1 refactor report

### Project References

- **[AGENTS.md](AGENTS.md)** — Engineering agent guardrails, Big 8 coverage rules, naming conventions
- **[DATA_LINKS_MASTER.md](DATA_LINKS_MASTER.md)** — Canonical data source URLs and API documentation
- **[config/data_sources.yaml](config/data_sources.yaml)** — Machine-readable data source configurations
- **[config/bucket_feature_selectors.yaml](config/bucket_feature_selectors.yaml)** — Big 8 bucket feature mappings

---

## 🧑‍💻 Engineering Agent Prompt (Codex/Cursor)

Use this developer prompt when starting a Codex/Cursor session or major change:

```text
You are the CBI-V15 Engineering Agent.

Follow the system rules and Cursor rules.json.

Task:
I want you to operate strictly within the CBI-V15 architecture.
Before making any changes:
1. Validate context.
2. If any file or directory is missing, ask me for it.
3. Explain your plan BEFORE writing code.
4. Produce minimal, surgical diffs.

Never hallucinate imports, modules, directories, dependencies, or data sources.
Never reintroduce BigQuery or v14 patterns.
Never write code outside the defined directories.
Keep everything aligned with the V15.1 training engine: Big 8 Tabular → Core TS → Meta → Ensemble → Monte Carlo.

When ready, ask: "Show me the files involved in this operation."
```

---

## 📦 Big 8 Bucket Modeling Rules

Enforce these rules in all engineering plans and changes:

- Big 8 buckets are: Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility
- Bucket features are derived from SQL macros only in `database/macros/`
- Use AutoGluon `TabularPredictor` for all Big 8 bucket specialists
- Use AutoGluon `TimeSeriesPredictor` for core ZL forecasting
- Meta model fuses Big 8 + core ZL outputs
- Ensemble layer smooths predictions into final forecasts
- Monte Carlo simulation produces probabilistic scenarios (VaR/CVaR), not raw forecasts

---

## 🧭 Notes & Conventions

- Ingestion lives under `trigger/<Source>/Scripts/` (no src/ingestion for new work).
- Features in SQL macros only (`database/macros/`); avoid Python feature loops.
- Training on Mac M4 CPU; `presets='extreme_quality'` (slower without GPU).
- Keep repo clean: configs/ignores inside their folders (`.cursor/*`, `.kilocode/*`, `augment/*`); no stray files at root.

---

## 📝 License

MIT — see [LICENSE](LICENSE).
