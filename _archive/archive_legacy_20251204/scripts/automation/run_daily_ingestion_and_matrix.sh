#!/bin/zsh
# Daily ingestion + matrix rebuild
# - Runs once per day on the Mac (e.g., 02:00 local)
# - Updates FRED buckets, Databento daily OHLCV, and rebuilds training.daily_ml_matrix

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting daily ingestion + matrix rebuild..."

# FRED buckets (FX, rates/curve, financial conditions)
python src/ingestion/fred/collect_fred_fx.py
python src/ingestion/fred/collect_fred_rates_curve.py
python src/ingestion/fred/collect_fred_financial_conditions.py

# Databento daily / hourly backfill (safe to re-run)
python scripts/ingestion/databento/pull_missing_symbols.py

# Optional: EIA / weather / USDA (will no-op if keys/series not yet configured)
python src/ingestion/eia/collect_eia_biofuels.py || echo "EIA biofuels ingestion skipped (missing key or series)."
python src/ingestion/weather/collect_weather_noaa.py || echo "NOAA ingestion skipped (permissions)."

# Rebuild staging.market_daily from raw.databento_futures_ohlcv_1d
python - << 'PY'
from google.cloud import bigquery

client = bigquery.Client(project="cbi-v15")
sql = """
CREATE OR REPLACE TABLE `cbi-v15.staging.market_daily`
PARTITION BY DATE_TRUNC(date, MONTH)
CLUSTER BY symbol, date AS
SELECT date, symbol, open, high, low, close, volume, open_interest
FROM `cbi-v15.raw.databento_futures_ohlcv_1d`;
"""
job = client.query(sql, location="us-central1")
job.result()
print("✅ Rebuilt staging.market_daily")
PY

# Rebuild and load training.daily_ml_matrix
python src/features/build_daily_ml_matrix.py
python scripts/load_daily_ml_matrix.py \
  --parquet TrainingData/exports/daily_ml_matrix.parquet \
  --table cbi-v15.training.daily_ml_matrix \
  --write-disposition WRITE_TRUNCATE

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Daily ingestion + matrix rebuild complete."

