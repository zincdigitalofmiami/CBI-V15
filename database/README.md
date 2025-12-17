# Database

> V15.1 Database Schema for MotherDuck + Local DuckDB
>
> **Read first:** `docs/architecture/MASTER_PLAN.md`, `AGENTS.md`, active plan in `.cursor/plans/`

All SQL schema definitions, macros, seeds, and tests for CBI-V15 (DuckDB/MotherDuck).

## Structure

```text
database/
├── ddl/                    # DDL files (CREATE TABLE statements)
│   ├── 00_schemas.sql      # Creates all 9 schemas
│   ├── 01_reference/       # Reference/lookup tables
│   ├── 02_raw/             # Raw vendor tables
│   ├── 03_staging/         # Cleaned/normalized data
│   ├── 04_features/        # Engineered features
│   ├── 05_training/        # Training artifacts
│   ├── 06_forecasts/       # Serving contract (dashboard reads here)
│   ├── 07_ops/             # Operations/monitoring
│   └── 08_explanations/    # SHAP/feature importance
├── macros/                 # SQL macros (reusable functions)
│   ├── big8_bucket_features.sql    # Dashboard SCORES (0-100)
│   ├── bucket_features.sql         # Model FEATURES (vectors)
│   ├── technical_indicators_all_symbols.sql
│   ├── cross_asset_features.sql
│   └── ...
├── seeds/                  # Reference data seeding
│   ├── seed_symbols.py     # 33 canonical symbols
│   ├── seed_regimes.py     # Regime weights
│   └── seed_splits.py      # Train/val/test splits
├── migrations/             # Versioned migrations
│   ├── versions/
│   │   └── V0001__init.sql
│   └── migrate.py
└── tests/                  # SQL tests
    ├── sql/
    │   ├── test_schemas_exist.sql
    │   ├── test_no_future_leakage.sql
    │   └── ...
    └── harness.py
```

## Schemas (9 total)

| Schema         | Purpose                                      | Table Count |
| -------------- | -------------------------------------------- | ----------- |
| `raw`          | Immutable source data from collectors        | 12          |
| `staging`      | Cleaned/normalized daily grains              | 7           |
| `features`     | Engineered features, bucket materializations | 10+         |
| `features_dev` | Dev views (pre-snapshot)                     | varies      |
| `training`     | OOF predictions, meta matrices, weights      | 6           |
| `forecasts`    | Serving contract (dashboard reads this)      | 4           |
| `reference`    | Symbols, calendars, regimes, driver maps     | 10          |
| `ops`          | Ingestion status, alerts, data quality       | 7           |
| `explanations` | SHAP values (weekly)                         | 1           |

## Data Flow

```
Trigger.dev jobs → raw.* (MotherDuck, source of truth)
                       ↓
              AnoFox SQL macros
                       ↓
              staging.* → features.*
                       ↓
         [SYNC] → Local DuckDB (training)
                       ↓
         AutoGluon training (Mac M4)
                       ↓
              predictions → forecasts.*
                       ↓
              Vercel Dashboard
```

## Runtime Topology

- **MotherDuck** = Cloud source of truth. All ingestion writes here.
- **Local DuckDB** = Ephemeral compute for training. Attaches MotherDuck via:

  ```sql
  -- Local DuckDB session (main = local scratch)
  -- Attach MotherDuck by name (NO alias in workspace mode)
  ATTACH 'md:usoil_intelligence?motherduck_token=${MOTHERDUCK_TOKEN}';

  -- Verify attachment
  SHOW DATABASES;

  -- Query cloud data using database.schema.table
  SELECT * FROM usoil_intelligence.features.daily_ml_matrix_zl;
  ```

Same schemas work in both environments. The separation is **runtime namespace**, not separate DDL files.

## Deployment

```bash
# Create/refresh schemas
python scripts/setup_database.py --both

# Run migrations
python database/migrations/migrate.py

# Seed reference data
python database/seeds/seed_reference_tables.py

# Sync cloud → local
python scripts/sync_motherduck_to_local.py

# Run tests
python database/tests/harness.py
```

## Big 8 Bucket Modeling Rules

The Big 8 buckets are:

1. **Crush** - ZL/ZS/ZM spread economics
2. **China** - China demand proxy
3. **FX** - Currency effects
4. **Fed** - Monetary policy
5. **Tariff** - Trade policy
6. **Biofuel** - RIN prices, biodiesel
7. **Energy** - Crude, HO, RB
8. **Volatility** - VIX, stress indices

Rules:

- Bucket features implemented via SQL macros in `database/macros/`
- `bucket_features.sql` = model FEATURES (vectors for training)
- `big8_bucket_features.sql` = dashboard SCORES (0-100 for display)
- AutoGluon `TabularPredictor` for specialists
- AutoGluon `TimeSeriesPredictor` for core ZL
- Meta model fuses Big 8 + core outputs

## 33 Canonical Symbols

- **Agricultural (11):** ZL, ZS, ZM, ZC, ZW, KE, ZO, CT, KC, SB, CC
- **Energy (4):** CL, HO, RB, NG
- **Metals (5):** GC, SI, HG, PA, PL
- **Treasuries (3):** ZN, ZB, ZF
- **FX (9):** 6A, 6B, 6C, 6E, 6J, 6M, 6N, 6S, DX
- **Palm (1):** FCPO

## Naming Conventions

| Concept                         | Pattern                         | Examples                         |
| ------------------------------- | ------------------------------- | -------------------------------- |
| **Volatility** (price variance) | `volatility_*`                  | `volatility_zl_21d`              |
| **Volume** (trading activity)   | `volume_*`                      | `volume_zl_sma_20`               |
| **Features**                    | `{source}_{symbol}_{indicator}` | `databento_zl_close`, `fred_dxy` |

**NEVER use `vol_*` alone** - always spell out `volatility_` or `volume_`.
