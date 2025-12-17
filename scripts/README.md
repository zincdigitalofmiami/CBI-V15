# Scripts

> Fast-moving workspace: review the latest `docs/architecture/MASTER_PLAN.md`, `AGENTS.md`, and active master plan `.cursor/plans/ALL_PHASES_INDEX.md` before running or editing scripts. Keep explorer clean; avoid duplicate helpers.

## Purpose

Operational and setup utilities (not core app code). Use these for setup, sync, validation, status.

## What Belongs Here

- `setup/` – env/setup/install (e.g., `install_autogluon_mac.sh`, schema deploy)
- `ops/` – health/status/connection checks (`system_status.sh`, `test_connections.py`)
- `export/` – data export utilities
- `deployment/` – deploy helpers
- `optimization/` – perf tuning scripts

## What Does NOT Belong Here

- Ingestion (→ `trigger/<Source>/Scripts/`)
- Feature engineering (→ `database/macros/`)
- Training code (→ `src/training/`)

## Key Scripts (high-use)

- `scripts/sync_motherduck_to_local.py` — sync MotherDuck → local DuckDB
- `scripts/setup/install_autogluon_mac.sh` — Mac M4 libomp + AutoGluon setup
- `scripts/validation/verify_core_macro_fx.py` — core feature/Terms-of-Trade QA
- `scripts/system_status.sh` — quick status sweep

## Naming Convention

`{action}_{subject}.py` or `{action}_{subject}.sh` (e.g., `check_data_availability.py`, `test_connections.py`)
