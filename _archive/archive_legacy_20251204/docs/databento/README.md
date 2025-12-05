---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**  
This project uses ONLY real, verified data sources. NO placeholders, synthetic data, NO fake values.  
All data must come from authenticated APIs, official sources, or validated historical records.
---

# Databento Integration – Index

**Purpose**: Central index for all Databento‑related docs and loaders in CBI‑V15.  
**Scope**: Venue‑pure CME/CBOT/NYMEX/COMEX futures via GLBX.MDP3 feeding BigQuery RAW → Python feature builders → Mac training.

---

## 1. Guides (`docs/databento/guides/`)

- `docs/databento/guides/INGESTION_WORKFLOW.md` (TBD)  
  - Will describe end‑to‑end Databento ingestion: key pairs, symbols, schedule, monitoring.

For now, see:
- `docs/architecture/MASTER_PLAN.md` – Databento as the canonical ZL futures source.  
- `docs/architecture/FINAL_COMPLETE_BQ_SCHEMA.sql` – `market_data.databento_futures_ohlcv_1m/1d` schema and how it feeds features/training.

---

## 2. Loaders (`docs/databento/loaders/`)

Python loaders live under `src/ingestion/databento/` and write directly to BigQuery RAW.

- `src/ingestion/databento/collect_daily.py`  
  - Daily OHLCV futures ingestion (ZL + core symbols) → `raw.databento_futures_ohlcv_1d`.  
  - Uses `DATABENTO_API_KEY` from macOS Keychain or Secret Manager (see `src/cbi_utils/keychain_manager.py`).

Supporting utilities:
- `scripts/ingestion/test_connections.py` – Verifies Databento key and connectivity.  
- `scripts/system_status.sh` – Shows Databento key status and basic raw‑table row counts.

---

## 3. Scheduler & Config

- `config/schedulers/ingestion_schedules.yaml`  
  - Defines Databento ingestion jobs (`databento_zl_price`, `databento_other_symbols`) for Cloud Scheduler.  
- `scripts/deployment/create_cloud_scheduler_jobs.sh`  
  - Example `gcloud scheduler jobs create http ...` commands for Databento ingestion functions.

---

## 4. How It Fits the Training Pipeline

Databento provides the **market_data backbone**:

1. `src/ingestion/databento/collect_daily.py` → `raw.databento_futures_ohlcv_1d`.  
2. Python feature builders (`src/features/build_daily_ml_matrix.py`) aggregate Databento prices, FX, regimes, etc. into `TrainingData/exports/daily_ml_matrix.parquet`.  
3. `scripts/load_daily_ml_matrix.py` loads the parquet into `training.daily_ml_matrix` (MONTH partitioned, clustered).  
4. Mac M4 training/export → forecasts written to `forecasts.zl_predictions_*`.  
5. `api.vw_latest_forecast` exposes the latest ZL forecast for the dashboard.

Heavy Databento features (microstructure, MBP‑10, intraday curves) can be added later via new feature tables without breaking this structure.

---

## 6. External Databento Docs (Authoritative References)

Use these when you’re wiring or extending loaders. They are the canonical source of truth for symbology, limits, schemas, and examples.

**API reference hubs**
- API reference index (historical/live/reference):  
  - https://databento.com/docs/api-reference-reference?historical=python&live=python&reference=python

**Historical API basics**
- Overview: https://databento.com/docs/api-reference-historical/basics/overview?historical=python&live=python&reference=python  
- Symbology: https://databento.com/docs/api-reference-historical/basics/symbology?historical=python&live=python&reference=python  
- Rate limits: https://databento.com/docs/api-reference-historical/basics/rate-limits?historical=python&live=python&reference=python  
- Size limits: https://databento.com/docs/api-reference-historical/basics/size-limits?historical=python&live=python&reference=python  

**Reference API basics**
- Symbology: https://databento.com/docs/api-reference-reference/basics/symbology?historical=python&live=python&reference=python  
- Dates and times: https://databento.com/docs/api-reference-reference/basics/dates-and-times?historical=python&live=python&reference=python  
- Errors: https://databento.com/docs/api-reference-reference/basics/errors?historical=python&live=python&reference=python  

**Venues, normalization, schemas**
- GLBX.MDP3 venue/dataset spec: https://databento.com/docs/venues-and-datasets/glbx-mdp3?historical=python&live=python&reference=python  
- Normalization standards: https://databento.com/docs/standards-and-conventions/normalization?historical=python&live=python&reference=python  
- “What’s a schema?”: https://databento.com/docs/schemas-and-data-formats/whats-a-schema?historical=python&live=python&reference=python  
- Databento binary encoding (DBN): https://databento.com/docs/standards-and-conventions/databento-binary-encoding?historical=python&live=python&reference=python  

**Examples – futures, ML, indicators**
- Futures introduction: https://databento.com/docs/examples/futures/futures-introduction?historical=python&live=python&reference=python  
- Options examples: https://databento.com/docs/examples/options?historical=python&live=python&reference=python  
- Algo trading – machine learning: https://databento.com/docs/examples/algo-trading/machine-learning?historical=python&live=python&reference=python  
- Algo trading – pairs trading (overview): https://databento.com/docs/examples/algo-trading/pairs-trading?historical=python&live=python&reference=python  
- Algo trading – CME WTI CL vs ICE Brent BRN pairs: https://databento.com/docs/examples/algo-trading/pairs-trading/cme-wti-cl-vs-ice-brent-crude-oil-brn?historical=python&live=python&reference=python  
- Basics – joining schemas: https://databento.com/docs/examples/basics-historical/joining-schemas?historical=python&live=python&reference=python  
- Basics – technical indicators: https://databento.com/docs/examples/basics-historical/technical-indicators?historical=python&live=python&reference=python  
- Basics – programmatic batch download: https://databento.com/docs/examples/basics-historical/programmatic-batch-download?historical=python&live=python&reference=python  


---
