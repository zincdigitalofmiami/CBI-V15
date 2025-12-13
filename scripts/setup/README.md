# Setup Scripts

## Purpose
Initial project setup and schema deployment scripts.

## What Belongs Here
- `execute_motherduck_schema.py` - Deploy SQL schemas to MotherDuck
- `store_api_keys.sh` - Store API keys in Keychain
- Environment initialization scripts

## What Does NOT Belong Here
- Production deployment (→ `scripts/deployment/`)
- Application code (→ `src/`)

## Key Scripts

### execute_motherduck_schema.py
Deploys all SQL from `database/models/` to MotherDuck in order:
1. 00_init → Create schemas
2. 01_raw → Raw tables
3. 02_staging → Staging tables
4. 03_features → Feature tables
5. 04_training → Training tables
6. 05_assertions → Data quality assertions
7. 06_api → API views

### Usage
```bash
export MOTHERDUCK_TOKEN="your_token"
export MOTHERDUCK_DB="cbi_v15"
python scripts/setup/execute_motherduck_schema.py
```

