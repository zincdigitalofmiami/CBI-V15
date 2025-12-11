# CBI-V15 Intelligence Platform

> **Institutional-grade ZL futures forecasting system** combining AI-driven orchestration with high-performance SQL-native feature engineering.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MotherDuck](https://img.shields.io/badge/Database-MotherDuck-blue)](https://motherduck.com)
[![Next.js](https://img.shields.io/badge/Dashboard-Next.js%2014-black)](https://nextjs.org)

---

## 🎯 Overview

CBI-V15 is a quantitative forecasting platform for ZL (Soybean Oil) futures that combines:

- **TSci Agents** — LLM-driven agentic orchestrator for experiment planning and model selection
- **AnoFox Engine** — High-performance SQL-native feature engineering within DuckDB
- **Next.js Dashboard** — Real-time visualization and intelligence reporting
- **MotherDuck** — Cloud-native data warehouse for production forecasts

**Key Innovation**: TSci acts as the "Brain" (strategic decision-making) while AnoFox acts as the "Muscle" (fast SQL feature computation), creating a hybrid system optimized for both intelligence and performance.

---

## ✨ Key Features

| Feature                | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| 🧠 **TSci Agents**     | OpenAI-powered orchestration (Curator, Planner, Forecaster, Reporter) with hallucination guardrails |
| ⚡ **AnoFox Engine**   | SQL-native feature engineering with 300+ features across 38 symbols |
| 📊 **Big 8 Drivers**   | Crush, China, FX, Fed, Tariff, Biofuel, Energy, **Volatility** (focus overlays, not cages) |
| 🎯 **Multi-Model**     | LightGBM, CatBoost, XGBoost quantile models with AutoML sweeps |
| 📈 **QRA Ensemble**    | Regime-weighted Quantile Regression Averaging (L3) |
| 🎲 **Monte Carlo**     | 1,000-path risk simulation with VaR/CVaR/downside metrics (L4) |
| 🦆 **MotherDuck**      | Cloud data warehouse with local DuckDB mirroring                  |
| 📉 **TradingView**     | Live ZL charts, Forex Heatmap, and Tech Gauges (Dark Mode)        |
| 🎛️ **Regime-Aware**    | Adaptive models with TSci meta-learning framework                        |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources (38+ symbols)           │
│  Databento  │  ScrapeCreator  │  FRED  │  EIA  │ USDA  │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│      Ingestion Layer (trigger/<Source>/Scripts/)         │
│  • DataBento   • ScrapeCreators   • FRED   • EIA_EPA    │
│  • USDA        • CFTC             • Weather/NOAA        │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│          AnoFox Engine (src/engines/anofox/)             │
│  • build_features.py (300+ features, all symbols)       │
│  • build_training.py (train/val/test splits)           │
│  • anofox_bridge.py (TSci ↔ SQL interface)             │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│               MotherDuck (database/)                     │
│  raw → staging → features → training → forecasts        │
│  (8 schemas, 30+ tables, SQL macros, assertions, API)   │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│     TSci Agents + OpenAI (src/models/tsci/)              │
│  • curator.py    (data QA + LLM quality analysis)       │
│  • planner.py    (model selection + LLM suggestions)    │
│  • forecaster.py (QRA ensemble + LLM weighting)         │
│  • reporter.py   (narrative generation + LLM reports)   │
│  • model_sweep.py (AutoML-lite per bucket/horizon)      │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│        4-Level Model Stack (L1→L2→L3→L4)                 │
│  L1: Base Models (LightGBM, CatBoost, XGBoost)          │
│  L2: Meta-Learner (model_sweep.py, regime tagging)     │
│  L3: QRA Ensemble (regime-weighted quantile averaging)  │
│  L4: Monte Carlo (1,000 paths, VaR/CVaR, scenarios)    │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│               Next.js Dashboard (dashboard/)             │
│  • /forecasts   • /neural-quant  • /sentiment           │
│  • /market-overview  • /quant-admin (TSci reports)      │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
CBI-V15/
├── dashboard/            # 🌐 Next.js Dashboard
│   ├── app/              # App Router pages
│   └── components/       # Visualizations (TradingView, Nivo)
│
├── database/             # 🗄️ SQL Schemas & Macros
│   ├── definitions/      # 00-08 DDL files
│   └── macros/           # Feature SQL macros
│
├── src/                  # 🐍 Python Source
│   ├── engines/          # AnoFox engine + engine registry
│   ├── models/           # TSci agents (Curator, Planner, Forecaster, Reporter)
│   ├── ingestion/        # Data collectors (databento, fred, eia, scrape_creator, etc.)
│   ├── training/         # Baseline models (lightgbm, catboost, xgboost)
│   ├── ensemble/         # L3: QRA ensemble
│   ├── simulators/       # L4: Monte Carlo risk simulation
│   └── utils/            # OpenAI client, keychain manager
│
├── docs/                 # 📚 Documentation
│   ├── architecture/     # System design
│   └── project_docs/     # Migrated docs
│
├── scripts/              # 🔧 Utility Scripts
│
└── config/               # ⚙️ Configuration
    └── requirements/     # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- MotherDuck account

### 1. Clone Repository

```bash
git clone https://github.com/zincdigitalofmiami/CBI-V15.git
cd CBI-V15
```

### 2. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements/requirements.txt
```

### 3. Configure Environment

```bash
export MOTHERDUCK_DB=cbi_v15
export MOTHERDUCK_TOKEN=<your-token>
export SCRAPECREATOR_API_KEY=<your-key>
export FRED_API_KEY=<your-key>
```

### 4. Initialize Database

```bash
python scripts/setup/execute_motherduck_schema.py
```

### 5. Start Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## 🔑 Environment Variables

| Variable                | Description              | Required |
| ----------------------- | ------------------------ | -------- |
| `MOTHERDUCK_DB`         | MotherDuck database name (`cbi_v15`) | ✅       |
| `MOTHERDUCK_TOKEN`      | MotherDuck auth token    | ✅       |
| `OPENAI_API_KEY`        | OpenAI API key (for TSci agents) | ✅       |
| `OPENAI_MODEL`          | OpenAI model ID (default: `gpt-5.1`) | Optional |
| `DATABENTO_API_KEY`     | Databento API key        | ✅       |
| `FRED_API_KEY`          | FRED API key             | ✅       |
| `SCRAPECREATOR_API_KEY` | ScrapeCreator API key    | ✅       |
| `EIA_API_KEY`           | EIA API key              | Optional |
| `USDA_NASS_API_KEY`     | USDA NASS API key        | Optional |

> Secrets: keep tokens/keys in a local `.env` (already gitignored), direnv, or macOS Keychain. Use `MOTHERDUCK_DB` (not `MOTHERDUCK_DATABASE`) set to your actual database name (default `cbi_v15`). Avoid committing shell init files with secrets.

---

## 📚 Documentation

- [V15 Architecture](docs/architecture/) — System design and data flow
- [Big 8 Drivers](docs/project_docs/BIG_8_DRIVERS.md) — Key market indicators
- [Feature Catalog](docs/project_docs/COMPLETE_FEATURE_LIST_290.md) — Complete feature list
- [TSci + AnoFox Integration](docs/project_docs/ANOFOX_TSCI_INTEGRATION.md) — How they work together

---

## 🛠️ Development

### Run Ingestion

```bash
python trigger/ScrapeCreators/Scripts/collect_news_buckets.py
python trigger/FRED/Scripts/collect_fred_fx.py
```

### Build Features & Training Data

```bash
# Build all features (300+ across 38 symbols)
python src/engines/anofox/build_features.py

# Build training tables with targets and splits
python src/engines/anofox/build_training.py
```

### Train Models

```bash
# Train baseline models (quantile regression: P10/P50/P90)
python src/training/baselines/lightgbm_zl.py
python src/training/baselines/catboost_zl.py
python src/training/baselines/xgboost_zl.py

# Or run TSci-orchestrated sweep
python src/models/tsci/planner.py
```

### Run Dashboard Locally

```bash
cd dashboard && npm run dev
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [MotherDuck](https://motherduck.com) — Cloud DuckDB
- [Databento](https://databento.com) — Market data
- [Vercel](https://vercel.com) — Dashboard hosting

---

## ⚡ About Zinc Digital

**Institutional Quantitative Architecture & AI Strategy**

Building high-performance trading infrastructure and agentic forecasting engines involved in the global markets.

🌐 **[www.zincdigital.co](https://www.zincdigital.co)**

> _14 hour days, all hustle. Straight outta Miami._ 🌴

<br />

<div align="center">
  <p>Made with ❤️ by <a href="https://www.zincdigital.co">Zinc Digital</a></p>
</div>
