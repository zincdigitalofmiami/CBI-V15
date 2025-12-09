# AI Agents Workspace Instructions (CBI-V15)

## Read First
- Start with `README.md` (project overview, architecture, quick start)
- Follow the rules in `.github/copilot-instructions.md` (current architecture, conventions, anti-patterns)
- For feature engineering context, skim `database/README.md` (8-schema structure, AnoFox SQL macros)

## Non-Negotiables
- No fake or placeholder data; only authenticated, real sources.
- MotherDuck/DuckDB for database (no Google/BigQuery/Dataform).
- AnoFox SQL for feature engineering (Python wrappers in `src/features/`).
- Training happens locally on Mac M4 only; never in the cloud.
- Keep configs in YAML/JSON; never hardcode secrets. API keys live in `.env` (local) or macOS Keychain.
- Avoid costly resources (> $5/month) without explicit approval.

## Workspace Defaults
- Project structure: DuckDB SQL → `database/definitions/`, Python → `src/`, ops scripts → `scripts/`, configs → `config/`, docs → `docs/`.
- Run before declaring work done:
  - `python scripts/setup_database.py --both`
  - `bash scripts/system_status.sh`
- Primary target: ZL (soybean oil). Keep Big 8 drivers covered (crush, China, FX, Fed, tariff, biofuel, energy, volatility).
- Prefer existing naming: `{source}_{driver}_{metric}` (e.g., `databento_zl_close`), volatility_* for price volatility, volume_* for trading volume.

## Big 8 Drivers (with correlations)
1. **Crush Spread** (+0.78) - Soybean oil premium to meal
2. **China Demand** (+0.65) - Import volumes, crush rates
3. **USD/BRL FX** (-0.72) - Brazil export competitiveness
4. **Fed Policy** (-0.58) - Rate changes impact commodity futures
5. **Tariff Regime** (-0.42) - Trade policy shocks
6. **Biofuel Mandates** (+0.51) - RFS2, biodiesel blend rates
7. **Energy Prices** (+0.68) - WTI crude, diesel costs
8. **Volatility** (+0.34) - Market uncertainty, VIX correlation

Hidden drivers: Weather (La Niña -0.29), Palm Oil (+0.61), Argentina Production (-0.38)

## Model Hierarchy (L1 → L4)
- **L1**: Base quantile regressors (LightGBM, CatBoost, XGBoost) at q10/q50/q90
- **L2**: Meta-learning AutoML sweep (Optuna hyperparameter search)
- **L3**: QRA ensemble (weighted by forecast skill, inverse RMSE)
- **L4**: Monte Carlo simulation (1,000 paths from L3 quantiles)

File pattern: `src/models/level_{1-4}/`

## Key Commands
```bash
# Setup database schemas
python scripts/setup_database.py --both

# Check system health
bash scripts/system_status.sh

# Training workflow
python src/training/train_pipeline.py --phase forecast --level 1
```

## What Goes Where
- **src/ingestion/** - Data collectors (Databento, EIA, USDA APIs)
- **src/features/** - Feature engineering logic (Python wrappers for AnoFox SQL)
- **src/training/** - Model training orchestration (TSci agents call this)
- **database/definitions/** - SQL feature definitions (AnoFox macros)
- **scripts/** - Ops utilities (deploy, health checks, NOT core logic)
- **dashboard/** - Next.js frontend (Vercel deployment)

## Anti-Patterns
❌ Don't create features in Python (use SQL macros in database/definitions/)
❌ Don't reference OHLCV columns (only Close prices exist)
❌ Don't mix volatility (price) with volume (CFTC position sizes)
❌ Don't write to raw schema (ingestion-only) or read from tsci schema (TSci-only)
❌ Don't generate placeholder code with "TODO" comments (implement fully or ask)

## Execution Environment
- Local: Mac M4 Ultra, 24/7 training runs
- Cloud: MotherDuck (DuckDB SaaS), Vercel (dashboard)
- Python 3.12, .venv/, Black formatter, Ruff linter

## Implementation Guardrails
- Check resources before creating tables/files; audit outputs after changes.
- Use DuckDB SQL for joins/transforms; keep pipelines idempotent and partitioned by `date`.
- No cloud training or AutoML/Vertex/BQML.
- Avoid hardcoded paths/endpoints; add config entries under `config/` and `.env`.
- Document complex reasoning inline (why, not what) and update adjacent docs when behavior changes.

## If You Need Context
- Dashboards/Next.js live in `dashboard/` (Vercel).
- Feature library and indicators are summarized in `database/definitions/` (AnoFox SQL macros).
- For ingestion or data coverage questions, start with `docs/data_sources/` and `DATA_SOURCES_MASTER.md`.

## When Unsure
- Pause, ask for clarification, or point to the exact doc section you need. Do not guess or invent data.
