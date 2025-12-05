# CBI‑V15 – AI Assistant Rules & Contracts

This file defines how AI assistants must work inside CBI‑V15. It complements `docs/architecture/MASTER_PLAN.md` and is considered a hard contract unless explicitly updated here.

---

## 1. Architecture & Data Flow

- **Python‑first**: All feature engineering and modeling happens in Python on the Mac.  
  - BigQuery is storage + views + dashboard read layer only.  
  - No BQML, no AutoML, no Vertex AI training jobs.
- **Pipeline spine** (daily frequency):
  - `raw.*` → `raw_staging.*` → `staging.*` → `features.*` → `training.daily_ml_matrix` → local training → `predictions.*` / `signals.*` → dashboard.
- **Canonical matrix**:
  - `training.daily_ml_matrix` is the single denormalized daily feature + target table.  
  - Training code must not perform joins at train time; everything needed must be columns in this matrix.

---

## 2. Datasets, Partitioning & Naming

- **BigQuery datasets** (all in `us-central1`):
  - `raw`, `raw_staging`, `staging`, `features`, `training`, `predictions`, `signals`, `reference`, `ops`, `archive`.
- **Partitioning**:
  - All long‑horizon daily tables must use monthly partitions:  
    - `PARTITION BY DATE_TRUNC(date, MONTH)`  
  - Cluster by logical key:  
    - e.g. `CLUSTER BY symbol` or `symbol, regime` or `series_id`.
- **Primary key contracts** (examples, must be respected in MERGEs):
  - `raw.databento_futures_ohlcv_1d`: `(symbol, date)`  
  - `raw.fred_economic`: `(series_id, date)`  
  - `raw.weather_noaa`: `(station_id, date, metric)`
- **Column prefixes** (only `date` and `symbol` are un‑prefixed):
  - FRED: `fred_*` (e.g. `fred_dgs2`, `fred_vixcls`)  
  - FX technicals: `fx_*` (e.g. `fx_brl_mom_21d`)  
  - Weather: `weather_{country}_{region}_{var}`  
  - EIA: `eia_*` (biofuels, RINs)  
  - USDA: `usda_*`  
  - CFTC: `cftc_*`  
  - ScrapeCreators: short prefix (e.g. `scrc_*`)  
  - Policy/Trump: `policy_trump_*`  
  - Volatility: `vol_*`
- **Naming bans**:
  - Do **not** introduce new table/column names containing “clean”, “v2”, “final”, “real” (or obvious variants) as suffixes.  
  - For staging, prefer `_panel`, `_aggregated`, `_daily`.

---

## 3. Ingestion & ETL Patterns

- **Standard pattern for ALL ingesters**:
  - Python script under `src/ingestion/<source>/collect_<source>_<bucket>.py`.  
  - Fetch external data → DataFrame → write to `raw_staging.<source>_<bucket>_<run_id>` with `WRITE_TRUNCATE`.  
  - `MERGE` from staging into canonical `raw.<source>_<domain>` using the declared PK.
- **Idempotence**:
  - Every ingestion must be safe to re‑run; re‑running for the same date range must not produce duplicates.  
  - MERGE on PK is mandatory; no blind `INSERT` into canonical `raw.*`.
- **Current sources (non‑exhaustive)**:
  - Databento: `raw.databento_futures_ohlcv_1d` (ZL, ZS, ZM, ZC, CL, HO, RB, GC, SI, HG, HE, etc.).  
  - FRED: `raw.fred_economic` (FX, rates, curve, macro, risk/credit) via the segmented collectors:
    - `collect_fred_fx.py`, `collect_fred_rates_curve.py`, `collect_fred_financial_conditions.py`.  
  - EIA: `raw.eia_biofuels` via `collect_eia_biofuels.py`.  
  - USDA: `raw.usda_reports` (planned) via USDA collectors.  
  - CFTC: `raw.cftc_cot` (planned) via `pycot‑reports`.  
  - Weather: `raw.weather_noaa` via `collect_weather_noaa.py`.  
  - ScrapeCreators: `raw.scrc_news_buckets` and related policy/trump buckets.

---

## 4. Provenance & Auditability

Any table an agent or ingestion script writes **must** include the following columns (at least in `raw.*`, and ideally propagated downstream):

- `source_url STRING` – where the data came from (API endpoint, URL, file path).  
- `fetched_at_utc TIMESTAMP` – exact UTC timestamp of retrieval.  
- `agent_version STRING` – semantic version of the script/agent, e.g. `cbi-v15@0.1.0`.  
- `content_hash STRING` – SHA‑256 hash of the raw payload bytes.  
- `run_id STRING` – unique ID per end‑to‑end run.

Rules:

- One `run_id` per pipeline run, passed along into downstream tables where feasible.  
- Hash the **raw** payload (before transforms).  
- Never overwrite canonical `raw.*` without going through staging + MERGE.  
- Use `run_id` and `agent_version` to explain changes in features and forecasts.

---

## 5. Feature Engineering Contracts

- **Canonical builder**:
  - `src/features/build_daily_ml_matrix.py` is the only source of the daily training matrix.  
  - It may call helper builders, but all flows into a single wide DataFrame → Parquet → `training.daily_ml_matrix`.
- **Matrix expectations**:
  - Contains all symbols needed for cross‑asset features (ZL, ZS, ZM, ZC, CL, HO, RB, GC, SI, HG, HE, etc.).  
  - No joins at train time: trainers operate directly on the matrix.
- **ZL is primary**:
  - Targets, baseline evaluations, and dashboard focus are on `symbol = 'ZL'`.  
  - Other symbols act as drivers/spreads/cross‑asset inputs.
- **Targets**:
  - Level targets: `target_{1w,1m,3m,6m,12m}` are **future price levels** (no returns).  
  - Training uses return targets derived from these levels in the training scripts.
- **Feature families that must be preserved and extended, not removed**:
  - Core price/vol: lags, returns, `vol_21d`, GK vol (`gk_vol_21d`), SMAs/EMAs, MA distances.  
  - FX techs: BRL & DXY momentum/vol, ZL–FX correlations, `terms_of_trade_zl_brl`.  
  - Macro FRED: `fred_*` levels for policy (DFF/DFEDTARU/DFEDTARL/EFFR/SOFR), curve (`DGS*`, `T10Y2Y`, `T10Y3M`), risk (`VIXCLS`, `NFCI`, `BAAFFM`, `BAMLH0A0HYM2`).  
  - Spreads: `boho_spread` (locked formula), crush margins, palm/rapeseed/corn spreads as they are added.  
  - Regime: `regime`, `regime_weight` from `reference.regime_calendar`.  
  - Advanced stats (as implemented): Hurst, Kalman slope, Amihud, fractal dimension, etc.

Assistants must **add** to this structure, not silently remove or rename existing features without updating this document and the MASTER_PLAN.

---

## 6. Modeling & SHAP Pruning

- **Baselines**:
  - LightGBM ZL baselines at horizons: `1w`, `1m`, `3m`, `6m`.  
  - Use return‑space targets; report MAE, RMSE, R², MAPE in price space.  
  - Goal: 1w test MAPE < 4%; 1m/3m/6m should improve as more drivers are wired.
- **SHAP pruning for 1m (institutional pattern)**:
  - Train a full 1m model on the widest reasonable feature set (≈250–400 columns).  
  - Compute SHAP values on the 1m validation split.  
  - Rank features by absolute mean SHAP.  
  - Keep features up to 99% cumulative SHAP gain.  
  - Drop features contributing <0.01% of total SHAP gain.  
  - Cluster SHAP patterns and keep the top feature per cluster (~70% correlation cutoff).  
  - Retrain on the pruned set and ensure validation MAE degradation ≤ 2%.
- **Script**:
  - `scripts/analysis/shap_prune_zl_1m.py` implements this logic and must be kept in sync with training outputs and paths.

---

## 7. Dashboard Expectations

- Training metrics:
  - API (e.g. `/api/training/metrics`) should summarize per‑horizon metrics from `predictions.*`.
- “What’s moving ZL” charts must be grounded in real math:
  - ZL price + MAs + vol + Sharpe.  
  - Rolling correlations: ZL vs palm/canola/corn, CL/HO, key FX (USDBRL, USDCNY, USDARS, majors), VIX, and key FRED macro.  
  - Fundamental spreads: crush margins, BOHO, palm–soy, canola–soy, etc.  
  - Macro curves: 2y/5y/10y yields and spreads vs time.  
  - SHAP driver charts: top features and family‑level SHAP contributions for the 1m model.  
  - Forecast overlays: historical ZL plus multi‑horizon forecasts with brief driver annotations.

---

## 8. Interaction & Guardrails

- Treat this project as if you’re working on a real GS/JPM/Citadel commodity pod:
  - Feature sets should be rich and cross‑asset (~200+ useful features), not “20 retail indicators.”  
  - No toy MACD‑only strategies; prefer statistical, cross‑asset, and fundamental drivers.
- **ZL context**:
  - BRL is the primary FX pair for ZL (USDBRL always matters).  
  - Palm and canola/rapeseed must eventually be ingested from non‑CME vendors for proper substitution spreads.
- Always:
  - Stay inside `us-central1`.  
  - Avoid creating new GCP resources without explicit user approval.  
  - Keep `MASTER_PLAN.md` and this file aligned with actual code and schema after major changes.

