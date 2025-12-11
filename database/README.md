# Database

This directory contains all SQL schema definitions, macros, and views for the CBI-V15 system.

## Structure

```text
database/
├── definitions/        # DDL files (AnoFox SQL macros-style organization)
│   ├── 00_init/       # Create schemas (raw, staging, features, etc.)
│   ├── 01_raw/        # Raw data tables (databento, fred, eia, scrapecreators, cftc, usda, weather)
│   ├── 02_staging/    # Staging tables (market, crush, china, news)
│   ├── 03_features/   # Feature tables (technical_indicators, daily_ml_matrix)
│   ├── 04_training/   # Training tables (with targets and splits)
│   ├── 05_assertions/ # Data quality assertions
│   └── 06_api/        # API views for dashboard
├── macros/            # SQL macros (reusable functions)
│   ├── features.sql                          # Price/return macros
│   ├── technical_indicators_all_symbols.sql  # RSI, MACD, BB, ATR (38 symbols)
│   ├── cross_asset_features.sql              # Correlations, spreads
│   ├── big8_cot_enhancements.sql             # CFTC COT helpers
│   ├── big8_bucket_features.sql              # Big 8 bucket scores
│   └── master_feature_matrix.sql             # Master builder (300+ features)
└── views/             # SQL views for internal analysis
```

## Schemas (8 Total)

| Schema     | Owner   | Purpose                          | Table Count |
|------------|---------|----------------------------------|-------------|
| `raw`        | Ingestion | Immutable source data (databento, fred, eia, etc.) | 15+ |
| `staging`    | Anofox  | Cleaned/normalized time series     | 10+ |
| `features`   | Anofox  | Engineered features (300+, Big 8 buckets, daily_ml_matrix) | 10+ |
| `training`   | Anofox  | ML training matrices (with targets, splits, regimes) | 5+ |
| `forecasts`  | TSci    | Model predictions (per horizon: 1w, 1m, 3m, 6m, 12m) | 5+ |
| `reference`  | System  | Catalogs, calendars, metadata | 5+ |
| `ops`        | System  | Ingestion status, pipeline metrics | 3+ |
| `tsci`       | TSci    | Agent jobs, runs, qa_checks, simulations | 4+ |

## V15 Contract

- **Anofox** builds features in `raw` → `staging` → `features` → `training` (via SQL macros).
- **TSci agents** orchestrate models, read from `training.*`, write forecasts to `forecasts.*` and logs to `tsci.*`.
- **Baselines** (LightGBM/CatBoost/XGBoost) train on `training.*` tables.
- **QRA + Monte Carlo** combine model outputs, write final forecasts to `forecasts.*`.

## Deployment

Deploy all schemas and macros to MotherDuck:
```bash
python scripts/setup_database.py --both
```

See `DATABASE_SETUP_GUIDE.md` for details.
