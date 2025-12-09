# CBI‑V15 Agents Workspace Guide

## Read First
- `SYSTEM_STATUS_COMPLETE.md` — comprehensive system reference (schemas, tables, models, data coverage)
- `.github/copilot-instructions.md` — agent guardrails, conventions, anti‑patterns
- `database/README.md` — 8‑schema layout, SQL macros, feature boundaries
- `CFTC_COT_INGESTION_COMPLETE.md` — CFTC COT pipeline reference

## ⛔ HARD STOP RULES (CRITICAL)
Before creating ANYTHING new, verify these conditions:

1. **NO NEW FILES** until existing work is complete:
   - All SQL macros in `database/macros/` must be tested and working
   - All ingestion scripts in `src/ingestion/` must successfully pull data
   - All existing feature tables must be populated with real data
   - Documentation must match actual implementation

2. **NO NEW FEATURES** until the 93+ existing features are built and validated:
   - Check `database/macros/master_feature_matrix.sql` for current inventory
   - Run `python scripts/setup_database.py --both` to verify schemas exist
   - If features are missing, **build them first** before proposing new ones

3. **NO NEW DOCUMENTATION** files. Period.
   - Update existing docs in place
   - If you need to document something, add it to an existing file
   - Delete temporary notes after use; do not accumulate markdown sprawl

4. **BEFORE ANY CHANGE**, verify current state:
   - What exists? (read the file/table/schema first)
   - What's missing? (compare to expected state)
   - What's broken? (run validation scripts)

## Non‑Negotiables
- No fake/placeholder data. Ask for missing schemas; do not guess.
- No Google/BigQuery/Dataform. Use DuckDB locally and MotherDuck in cloud.
- Feature engineering in SQL (AnoFox macros). Do not build features in Python.
- Close prices only; do not reference Open/High/Low/Volume for price features.
- Train locally on the Mac M4; no cloud training/Vertex/AutoML.
- Secrets via `.env` or macOS Keychain; never hardcode.
- Keep costs minimal; avoid paid resources without approval (> $5/mo).
- **Never create new markdown files** unless you plan to keep them permanently. If kept, file in the correct folder (`docs/architecture/`, `docs/ops/`, etc.) or delete after use.

## Big 8 Buckets (Official Names)
The Big 8 are **focus overlays**, NOT exclusive feature sets. Models see ALL features.

| Bucket | Name | What It Represents |
|--------|------|-------------------|
| 1 | **Crush** | ZL/ZS/ZM spread economics, oil share, board crush |
| 2 | **China** | China demand proxy (HG copper, export sales) |
| 3 | **FX** | Currency effects (DX, BRL, CNY, MXN) |
| 4 | **Fed** | Monetary policy (Fed funds, yield curve, NFCI) |
| 5 | **Tariff** | Trade policy (Trump sentiment, Section 301) |
| 6 | **Biofuel** | RIN prices, biodiesel, RFS mandates, BOHO spread |
| 7 | **Energy** | Crude, HO, RB, crack spreads |
| 8 | **Volatility** | VIX, realized vol, STLFSI4, stress indices |

## Naming Conventions (MANDATORY)

### Volatility vs Volume (CRITICAL)
| Concept | Pattern | Examples | NEVER USE |
|---------|---------|----------|-----------|
| **Volatility** (price variance) | `volatility_*` | `volatility_zl_21d`, `volatility_bucket_score` | `vol_*` alone |
| **Volume** (trading activity) | `volume_*` | `volume_zl_21d`, `open_interest_zl` | `vol_*` alone |

### Feature Naming
Pattern: `{source}_{symbol}_{indicator}_{param}_{transform}`
- ✅ `databento_zl_close`, `volatility_vix_close`, `cftc_zl_managed_money_net_pct`
- ❌ `vol_zl_21d` (ambiguous), `volat_regime` (inconsistent)

## Workspace Defaults
- Structure: SQL → `database/definitions/`, Python → `src/`, ops → `scripts/`, configs → `config/`, docs → `docs/`.
- Before calling work done, run:
  - `python scripts/setup_database.py --both`
  - `bash scripts/system_status.sh`
  - Optional: `python scripts/verify_pipeline.py`
- Primary target: ZL (soybean oil). Cover Big 8 drivers.
- 38 futures symbols: Agricultural (11), Energy (4), Metals (5), Treasuries (3), FX (10), plus FCPO (palm oil).

## Implementation Guardrails
- Check resources before creating; audit outputs after changes.
- Use DuckDB/MotherDuck SQL for joins/transforms; keep pipelines idempotent and date‑partitioned.
- Parameterize SQL; avoid f‑strings for queries. Handle errors explicitly.
- Keep Python as orchestration (TSci) and bridges only; heavy math/indicators live in SQL.
- Update adjacent docs when behavior changes (explain why, not just what).

## What Goes Where
- `src/ingestion/` — collectors (Databento, EIA, USDA, CFTC, FRED)
- `src/features/` — Python wrappers around AnoFox SQL macros
- `src/training/` — orchestration of L1–L4 model stack
- `database/definitions/` — schemas, feature tables, assertions
- `database/macros/` — reusable SQL feature macros (Big 8, technicals, spreads)
- `scripts/` — ops utilities (setup, status, deploy); not core logic

## If You Need Context
- Dashboard lives in `dashboard/` (Vercel). Queries read `forecasts.*` in MotherDuck.
- Data sources: see `DATA_LINKS_MASTER.md` and `WEB_SCRAPING_TARGETS_MASTER.md`.
- Integration details: `README.md`, `docs/project_docs/tsci_anofox_architecture.md`.

## When Unsure
- Pause and ask. Point to the exact doc section you need. Never invent data, columns, or paths.
