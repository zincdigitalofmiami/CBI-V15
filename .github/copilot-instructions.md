# CBI-V15: Institutional ZL Futures Forecasting Platform

## Architecture Overview
**TSci (Brain)** = Python/LLM agents orchestrate workflows (src/tsci/)
**AnoFox (Muscle)** = SQL feature engineering in DuckDB (database/definitions/)

Data flow: raw → staging → features → training → forecasts (8 schemas, see database/README.md)
Bridge: `src/engines/anofox_bridge.py` connects Python to DuckDB features

## Critical Conventions
- **NO HALLUCINATION**: Never generate fake data, fake columns, or fake file paths
- **Close prices only** (no OHLCV) - if code references Open/High/Low/Volume, it's wrong
- **Naming pattern**: `{source}_{driver}_{metric}` (e.g., `cftc_crush_net_long`)
  - Use `volatility_*` (not `volume_*` or `vol_*`) for price volatility
  - Use `volume_*` only for CFTC position volumes
- **Source prefixes**: cftc_, eia_, usda_, databento_, cme_
- **Schema boundaries**: AnoFox builds features (staging/features), TSci trains models (training/forecasts)

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
- **src/ingestion/** - Data collectors (CFTC, EIA, USDA APIs)
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
