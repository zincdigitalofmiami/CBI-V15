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
| 🧠 **TSci Agents**     | AI-powered experiment planning, model selection, and QA           |
| ⚡ **AnoFox Engine**   | SQL-native feature engineering with DuckDB macros                 |
| 📊 **Big 8 Drivers**   | Crush, China, FX, Fed, Tariff, Biofuel, Energy, Vol               |
| 🦆 **MotherDuck**      | Cloud data warehouse with local DuckDB mirroring                  |
| 📉 **TradingView**     | Live ZL charts, Forex Heatmap, and Tech Gauges (Dark Mode)        |
| 🔮 **Crystal Ball AI** | "Driver of Drivers" analysis for Lobbying, SAF, and Weather risks |
| 🎛️ **Regime-Aware**    | Adaptive models based on market conditions                        |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                         │
│  Databento  │  ScrapeCreator  │  FRED  │  EIA          │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│               Ingestion Layer (src/ingestion/)           │
│  • databento/   • scrape_creator/   • fred/   • eia/    │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│               AnoFox Engine (src/engines/anofox/)        │
│  • build_features.py   • build_training.py              │
│  • build_forecasts.py  • anofox_bridge.py               │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│               MotherDuck (database/)                     │
│  raw → staging → features → training → forecasts        │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│               TSci Agents (src/models/tsci/)             │
│  • planner.py   • curator.py   • forecaster.py          │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│               Next.js Dashboard (dashboard/)             │
│  • /forecasts   • /neural-quant  • /sentiment           │
│  • /market-overview  • /quant-admin                     │
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
│   ├── schema/           # 00-08 DDL files
│   └── macros/           # Feature SQL macros
│
├── src/                  # 🐍 Python Source
│   ├── engines/          # AnoFox engine
│   ├── models/           # TSci agents (Planner, Curator, Forecaster)
│   ├── ingestion/        # Data ingestion
│   └── training/         # Model training
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
export MOTHERDUCK_DB=cbi-v15
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
| `MOTHERDUCK_DB`         | MotherDuck database name | ✅       |
| `MOTHERDUCK_TOKEN`      | MotherDuck auth token    | ✅       |
| `SCRAPECREATOR_API_KEY` | ScrapeCreator API key    | ✅       |
| `FRED_API_KEY`          | FRED API key             | ✅       |
| `DATABENTO_API_KEY`     | Databento API key        | ✅       |
| `EIA_API_KEY`           | EIA API key              | Optional |

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
python src/ingestion/scrape_creator/collect.py
python src/ingestion/fred/collect_fred_fx.py
```

### Build Features

```bash
python src/engines/anofox/build_features.py
python src/engines/anofox/build_training.py
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
