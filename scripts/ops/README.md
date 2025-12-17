# Operations Scripts

## Purpose

Health checks, status monitoring, and connection testing. Run these to verify system health.

## What Belongs Here

- `check_data_availability.py` - Verify data freshness
- `ingestion_status.py` - Check ingestion pipeline status
- `test_connections.py` - Test API and database connections
- `vegas/` - Vegas-specific operational scripts

## What Does NOT Belong Here

- Data collectors (→ `trigger/<Source>/Scripts/`)
- Setup scripts (→ `scripts/setup/`)

## Usage

```bash
# Check all connections
python scripts/ops/test_connections.py

# Check data freshness
python scripts/ops/check_data_availability.py

# Get ingestion status
python scripts/ops/ingestion_status.py
```
