#!/bin/zsh
# Intraday ingestion runner (Mac cron/launchd friendly)
# - Updates Databento 1h OHLCV + statistics for all configured symbols
# - Intended to run every 15 minutes on the Mac

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting intraday ingestion..."

# 1h OHLCV + 1d backfill for any missing symbols
python scripts/ingestion/databento/pull_missing_symbols.py

# Statistics stream (last 1 day)
python src/ingestion/databento/collect_statistics.py 1

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Intraday ingestion complete."

