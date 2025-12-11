# Scripts

## Purpose
Operational and setup scripts. These are utilities, NOT the core application code.

## What Belongs Here
- `deployment/` - Deploy to Vercel, MotherDuck, etc.
- `export/` - Data export utilities
- `ops/` - Health checks, status monitoring, connection tests
- `optimization/` - Performance tuning scripts
- `setup/` - Initial setup, schema deployment

## What Does NOT Belong Here
- Data collectors (→ `trigger/<Source>/Scripts/`)
- Feature engineering code (→ `src/features/`)
- Training code (→ `src/training/`)

## Naming Convention
`{action}_{subject}.py` or `{action}_{subject}.sh`

Examples:
- `deploy_schema.py`
- `check_data_availability.py`
- `test_connections.py`
