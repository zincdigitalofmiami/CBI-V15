# CBI-V15 Intelligence Platform

> **Read first – fast-moving workspace.** Always check the latest work before editing: `docs/architecture/MASTER_PLAN.md`, `AGENTS.md`, `DATA_LINKS_MASTER.md`, and the active master plan `.cursor/plans/ALL_PHASES_INDEX.md`. Multiple agents work in parallel—plans can drift; verify current files, avoid duplicating scripts/MDs/folders, and keep the explorer clean.

> **ZL (Soybean Oil) forecasting system** — DuckDB/MotherDuck + AutoGluon 1.4 + Trigger.dev, with SQL-first features.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MotherDuck](https://img.shields.io/badge/Database-MotherDuck-blue)](https://motherduck.com)
[![Next.js](https://img.shields.io/badge/Dashboard-Next.js%2014-black)](https://nextjs.org)

---

## 🎯 Overview

CBI-V15 combines:
- **SQL-first features (AnoFox macros)** — 300+ features in `database/macros/`
- **AutoGluon 1.4** — Quantile Tabular/TimeSeries (P10/P50/P90) on Mac M4
- **Trigger.dev** — Ingestion by source under `trigger/<Source>/Scripts/`
- **MotherDuck + Local DuckDB** — Cloud source of truth; local mirror for fast training
- **Next.js Dashboard** — Queries MotherDuck directly

---

## 🏗️ Architecture (V15.1)
- **Ingestion**: Trigger.dev jobs write to MotherDuck (`raw.*`), organized per source (`trigger/DataBento`, `trigger/FRED`, `trigger/EIA_EPA`, `trigger/USDA`, `trigger/CFTC`, `trigger/ScrapeCreators`, `trigger/ProFarmer`, `trigger/UofI_Feeds`, `trigger/Weather`, etc.)
- **Features**: SQL macros (AnoFox) in `database/macros/` → `features.*` tables/views
- **Training**: Sync MotherDuck → local DuckDB (`data/duckdb/cbi_v15.duckdb`), train AutoGluon quantile models (bucket specialists + main ZL)
- **Forecasts**: Upload to MotherDuck (`forecasts.zl_predictions`); dashboard reads from MotherDuck
- **Risk**: Monte Carlo in `src/simulators/monte_carlo_sim.py` (VaR/CVaR)

---

## 📁 Project Structure (clean layout)
```
CBI-V15/
├── trigger/                # Trigger.dev ingestion per source (Scripts/ + README per source)
├── database/               # Schemas (raw→features→forecasts) + macros (AnoFox)
├── src/
│   ├── engines/anofox/     # AnoFox bridge
│   ├── training/           # Baselines; AutoGluon module to be created
│   ├── simulators/         # Monte Carlo
│   └── models/             # (currently empty; reserved for local model definitions)
├── docs/                   # Architecture & ops docs
├── scripts/                # Setup, sync, validation, ops
├── config/                 # YAML/requirements
├── dashboard/              # Next.js app
└── .cursor/.kilocode/      # Tool-specific configs (plans, ignores)
```

---

## 🚀 Quick Start (local dev)
1) **Python env**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r config/requirements/requirements.txt
```
2) **Node deps (dashboard)**
```bash
cd dashboard && npm install
```
3) **Set env (example)**
```bash
export MOTHERDUCK_DB=cbi_v15
export MOTHERDUCK_TOKEN=<token>
export DATABENTO_API_KEY=<key>
export FRED_API_KEY=<key>
```
4) **Sync cloud → local DuckDB before training**
```bash
python scripts/sync_motherduck_to_local.py --dry-run
```
5) **Run dashboard**
```bash
cd dashboard && npm run dev
```

---

## 🔑 Environment (minimum)
- `MOTHERDUCK_DB`, `MOTHERDUCK_TOKEN`
- `DATABENTO_API_KEY`, `FRED_API_KEY`
- (Optional) `SCRAPECREATOR_API_KEY`, `EIA_API_KEY`, `USDA_NASS_API_KEY`
- Secrets in `.env` or macOS Keychain; never committed.

---

## 📚 Key Docs
- `docs/architecture/MASTER_PLAN.md` — source of truth (V15.1 AutoGluon hybrid)
- `AGENTS.md` — guardrails, Big 8 coverage, naming rules
- `DATA_LINKS_MASTER.md` — canonical data sources
- `PHASE_0_EXECUTION_READY.md` — readiness checklist
- `.cursor/plans/ALL_PHASES_INDEX.md` — active master implementation plan (Phases 0–5)

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
