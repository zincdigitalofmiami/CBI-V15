# CBI‑V15 Agents Workspace Guide

## AI AGENT MASTER GUIDELINES (CBI-V15)

**Path:** `/Volumes/Satechi Hub/CBI-V15/AI_GUIDELINES.md`

Read in this order before any task:

1. `docs/architecture/MASTER_PLAN.md`
2. `AGENTS.md`
3. `database/README.md`
4. `AI_GUIDELINES.md`
5. Active plan in `.cursor/plans/*.plan.md`

Operating protocol:

- Verify paths/APIs before use; no unconfirmed imports.
- Prevent leakage/look-ahead; use train-only stats; set `random_state/seed`.
- Follow naming rules (`volatility_*` vs `volume_*`; never `vol_*`).
- Run sanity checks (shapes, tests/linters) and remove temp debug.

Workspace hygiene:

- Keep Augment configs in `augment/`; no data/code/docs there.
- Avoid root clutter; update existing docs; sync plan changes in `.cursor/plans/`.

## Read First

- `docs/architecture/MASTER_PLAN.md` — V15.1 AutoGluon hybrid architecture (UPDATED Dec 9, 2025)
- `docs/architecture/SYSTEM_STATUS_COMPLETE.md` — comprehensive system reference (schemas, tables, models, data coverage)
- `database/README.md` — 8‑schema layout, SQL macros, feature boundaries
- `docs/ingestion/CFTC_COT_INGESTION_COMPLETE.md` — CFTC COT pipeline reference

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
   - Keep explorer clean: stash config/ignore files in their homes (`.cursor/*`, `.kilocode/*`, `augment/*`); no stray files at repo root

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

| Bucket | Name           | What It Represents                                |
| ------ | -------------- | ------------------------------------------------- |
| 1      | **Crush**      | ZL/ZS/ZM spread economics, oil share, board crush |
| 2      | **China**      | China demand proxy (HG copper, export sales)      |
| 3      | **FX**         | Currency effects (DX, BRL, CNY, MXN)              |
| 4      | **Fed**        | Monetary policy (Fed funds, yield curve, NFCI)    |
| 5      | **Tariff**     | Trade policy (Trump sentiment, Section 301)       |
| 6      | **Biofuel**    | RIN prices, biodiesel, RFS mandates, BOHO spread  |
| 7      | **Energy**     | Crude, HO, RB, crack spreads                      |
| 8      | **Volatility** | VIX, realized vol, STLFSI4, stress indices        |

## Naming Conventions (MANDATORY)

### Volatility vs Volume (CRITICAL)

| Concept                         | Pattern        | Examples                                       | NEVER USE     |
| ------------------------------- | -------------- | ---------------------------------------------- | ------------- |
| **Volatility** (price variance) | `volatility_*` | `volatility_zl_21d`, `volatility_bucket_score` | `vol_*` alone |
| **Volume** (trading activity)   | `volume_*`     | `volume_zl_21d`, `open_interest_zl`            | `vol_*` alone |

### Feature Naming

Pattern: `{source}_{symbol}_{indicator}_{param}_{transform}`

- ✅ `databento_zl_close`, `volatility_vix_close`, `cftc_zl_managed_money_net_pct`
- ❌ `vol_zl_21d` (ambiguous), `volat_regime` (inconsistent)

## Workspace Defaults

- Structure: SQL → `database/models/`, Python → `src/`, ops → `scripts/`, configs → `config/`, docs → `docs/`.
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
- Keep Python as orchestration and bridges only; heavy math/indicators live in SQL.
- Update adjacent docs when behavior changes (explain why, not just what).

## What Goes Where

- `trigger/<Source>/Scripts/` — data collection (Databento, EIA, EPA, USDA, CFTC, FRED, UofI_Feeds)
- `src/training/autogluon/` — AutoGluon TabularPredictor + TimeSeriesPredictor wrappers
- `src/features/` — Python wrappers around AnoFox SQL macros
- `src/training/` — training orchestration (bucket specialists + main predictor)
- `database/models/` — schemas, feature tables, assertions
- `database/macros/` — reusable SQL feature macros (Big 8, technicals, spreads)
- `scripts/` — ops utilities (setup, status, deploy, sync); not core logic
- `data/duckdb/` — local DuckDB mirror (training landing pad)
- `data/models/` — AutoGluon model artifacts

## If You Need Context

- Dashboard lives in `dashboard/` (Vercel). Queries read `forecasts.*` in MotherDuck.
- Data sources: see `DATA_LINKS_MASTER.md` (canonical) and `trigger/WEB_SCRAPING_TARGETS_MASTER.md` (web scraping URLs).
- Integration details: `README.md`, `docs/architecture/MASTER_PLAN.md`.

## When Unsure

- Pause and ask. Point to the exact doc section you need. Never invent data, columns, or paths.

## AI Assistant Behavior (Augment / LLMs)

You are working in the `CBI-V15` repo. Follow these rules strictly:

- Obey instruction precedence: system instructions > `AGENTS.md` > repo architecture/design docs > everything else (examples, web, prior code). If anything conflicts, follow the higher-priority source and ignore the rest.
- Do not hallucinate:
  - Never invent schemas, tables, columns, symbols, or file paths.
  - Never fabricate data, examples, or mock records unless the user explicitly asks for synthetic data.
  - If something is missing or unclear, stop and ask the user instead of guessing.
- Respect “no new work before cleanup”:
  - Do not propose or create new files, new features, or new model families while there are known broken or missing pieces in existing ingestion, feature engineering, or training pipelines.
  - Prioritize fixing and validating what already exists over adding anything new.
- Follow feature and naming rules:
  - Keep feature engineering where the repo’s rules put it (SQL layer for features; Python only for orchestration or glue).
  - Use the existing naming conventions for volatility vs volume and for feature names; avoid introducing new naming patterns.
  - Do not repurpose existing feature names to mean something different from their documented intent.
- Stay within the documented architecture:
  - Respect the existing storage, ETL, and training stack; do not introduce new databases, clouds, schedulers, or model-serving systems without explicit user direction.
  - Do not silently switch to different tools or libraries if they conflict with the documented stack.
- Validation and safety:
  - Do not declare a pipeline, feature, or model done without also describing how it should be validated (data quality checks, schema checks, basic evaluation metrics) and ensuring those checks fit the existing patterns.
  - Prefer idempotent, restart-safe changes; avoid designs that require manual cleanup or one-off steps.
- Code style and scope:
  - Match the existing code style and structure; extend patterns instead of inventing new frameworks.
  - Keep changes minimal and localized to the user’s request; do not refactor unrelated parts of the codebase without being asked.
- When unsure:
  - Prefer asking targeted clarification questions over making hidden assumptions.
  - Explicitly call out any trade-offs, risks, or uncertainties instead of hiding them in the implementation.

## AI Assistant Plan Building (Augment Code / Cursor Agent)

When building implementation plans:

1. **Always Read First** (in this order):
   - `docs/architecture/MASTER_PLAN.md` — V15.1 architecture source of truth
   - This file (`AGENTS.md`) — current guardrails and conventions
   - `database/README.md` — 8-schema layout and feature boundaries
   - `.cursor/plans/ALL_PHASES_INDEX.md` — active master implementation plan (Phases 0–5)

2. **Check Existing State Before Planning**:
   - Run `ls -la scripts/` to see available operational scripts
   - Check `src/ingestion/` for existing data collectors
   - Review `database/macros/` for existing SQL feature macros
   - Verify `config/requirements/requirements.txt` has current dependencies

3. **Plan Structure Requirements**:
   - **Phase 0**: Critical infrastructure + bug fixes (highest priority)
   - **Phase 1-N**: Incremental feature additions (after Phase 0 complete)
   - Each task must specify:
     - Exact file paths (no placeholders)
     - Specific schema/table names (verify they exist in `database/models/`)
     - Data sources (verify against `DATA_LINKS_MASTER.md`)
     - Validation steps (how to test it works)

4. **Technology Stack Constraints** (NEVER deviate):
   - **Database**: DuckDB (local mirror), MotherDuck (cloud source of truth) — NO BigQuery, NO Postgres
   - **ML Framework**: AutoGluon 1.4 (TabularPredictor + TimeSeriesPredictor hybrid) — NO custom sklearn pipelines
     - TabularPredictor: `presets='extreme_quality'` for bucket specialists (Mitra, TabPFNv2, TabICL, TabM + tree models)
     - TimeSeriesPredictor: Chronos-Bolt zero-shot baseline (CPU-compatible on Mac M4)
     - Problem type: `quantile` for P10/P50/P90 (probabilistic forecasts)
   - **Feature Engineering**: SQL macros in `database/macros/` (AnoFox) — NO Python feature loops
   - **Orchestration**: Trigger.dev jobs in `trigger/<Source>/Scripts/` — NO Airflow, NO Prefect
   - **Training**: Mac M4 local CPU (reads from local DuckDB mirror) — NO cloud training, NO GPUs
   - **Dashboard**: Next.js/Vercel querying MotherDuck — NO separate API server

5. **Data Source Validation** (Critical):
   - Only use sources listed in `DATA_LINKS_MASTER.md`
   - Verify API keys exist in `.env` or macOS Keychain before planning ingestion
   - Check symbol availability (38 futures symbols documented)
   - Confirm data frequency (daily, weekly, monthly)

6. **Dependency Chain Planning**:
   - Phase 0 bugs MUST be fixed before adding new features
   - Data ingestion MUST work before feature engineering
   - Features MUST exist before training models
   - Models MUST train before ensemble
   - Example: EPA RIN prices → EIA biofuels features → Biofuel bucket specialist → Greedy ensemble

7. **Big 8 Bucket Coverage** (Required):
   Each plan must explicitly cover all 8 buckets with their data sources:
   1. **Crush**: Databento (ZL/ZS/ZM), NOPA, farmdoc Grain Outlook
   2. **China**: **Farm Policy News: Trade** (MANDATORY), USDA FAS Export Sales, farmdoc Trade Policy
   3. **FX**: FRED FX series, Databento (6L/DX)
   4. **Fed**: FRED rates/curve, Farm Policy News: Budget, farmdoc: Interest Rates
   5. **Tariff**: **Farm Policy News: Trade** (MANDATORY), ScrapeCreators Trump, farmdoc Gardner Policy
   6. **Biofuel**: **EPA RIN Prices (D3/D4/D5/D6)** (FREE, weekly), EIA, farmdoc RINs (Scott Irwin)
   7. **Energy**: EIA petroleum, Databento (CL/HO/RB)
   8. **Volatility**: FRED VIXCLS, Databento VIX, STLFSI4

8a. **Big 8 Bucket Modeling Rules** (Enforced for all plans):

- Big 8 buckets are: Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility
- Features for buckets are derived from SQL macros only (`database/macros/`), not Python loops
- Use AutoGluon `TabularPredictor` for all Big 8 bucket specialists
- Use AutoGluon `TimeSeriesPredictor` for core ZL forecasting
- Meta model fuses Big 8 + core ZL outputs
- Ensemble layer smooths predictions into final forecasts
- Monte Carlo simulation produces probabilistic scenarios (VaR/CVaR), not raw forecasts

8. **Validation Requirements**:
   Every planned feature/model must include:
   - How to verify data was ingested (`SELECT COUNT(*) FROM raw.{table}`)
   - How to test feature engineering (`SELECT * FROM features.{table} LIMIT 5`)
   - How to validate model output (`python scripts/validation/check_forecasts.py`)
   - Expected output format (schema, column names, row counts)

9. **File Naming Conventions** (Mandatory):
   - Ingestion: `src/ingestion/{source}/{action}.py` (e.g., `src/ingestion/epa/collect_rin_prices.py`)
   - Features: `database/macros/{domain}_features.sql` (e.g., `database/macros/biofuel_features.sql`)
   - Training: `src/training/autogluon/{model_type}.py` (e.g., `src/training/autogluon/bucket_specialist.py`)
   - Trigger jobs: `trigger/ingestion/{domain}/{source}_{action}.ts` (e.g., `trigger/ingestion/energy_biofuels/epa_rin_prices.ts`)

10. **Anti-Patterns to Avoid**:
    - ❌ Creating new markdown files (update existing docs in-place)
    - ❌ Hardcoding API keys in code (use `.env` or Keychain)
    - ❌ Building features in Python loops (use SQL macros)
    - ❌ Training models before data pipeline is working
    - ❌ Proposing BigQuery/Dataform/GCP resources
    - ❌ Inventing new symbols not in the 38-symbol list
    - ❌ Creating placeholders or mock data
    - ❌ Skipping validation steps

## Summary: How to Build a Proper Plan

```bash
# Step 1: Read context
cat docs/architecture/MASTER_PLAN.md
cat AGENTS.md
cat .cursor/plans/ALL_PHASES_INDEX.md

# Step 2: Verify current state
ls -la src/ingestion/
ls -la database/macros/
cat config/requirements/requirements.txt

# Step 3: Build plan following this structure:
## Phase 0: Critical Bugs (MUST complete first)
- [ ] Task 1: Fix {specific bug} in {exact file path}
      Validation: {how to verify fix}

## Phase 1: Data Ingestion (Dependency: Phase 0 complete)
- [ ] Task 2: Add {specific data source} to raw.{table_name}
      Validation: SELECT COUNT(*) FROM raw.{table_name}

## Phase 2: Feature Engineering (Dependency: Phase 1 complete)
- [ ] Task 3: Create {feature_name} in database/macros/{file}.sql
      Validation: SELECT * FROM features.{table_name} LIMIT 5

## Phase 3: Model Training (Dependency: Phase 2 complete)
- [ ] Task 4: Train {bucket_name} specialist using AutoGluon TabularPredictor
      Validation: python scripts/validation/check_model_artifacts.py

## Phase 4: Ensemble & Monte Carlo (Dependency: Phase 3 complete)
- [ ] Task 5: Combine predictions with greedy_ensemble.py
      Validation: SELECT * FROM forecasts.zl_predictions LIMIT 5
```

## CBI-V15 Engineering Agent Startup Prompt (Codex/Cursor)

Use this developer prompt for Codex/Cursor sessions when starting a new task or major change:

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
