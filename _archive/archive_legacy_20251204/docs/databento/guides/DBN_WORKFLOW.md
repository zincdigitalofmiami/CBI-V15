# Databento Binary Encoding (DBN) Workflow – Future Plan

**Status**: Design-only for now. This documents how we will use Databento Binary Encoding (DBN) once we turn on intraday / microstructure ingestion. The current V15 baseline still uses higher-level helpers for daily OHLCV only.

## 1. Why DBN Matters Here

- **Smaller payloads, faster pulls** for large historical ranges and multi-symbol queries (especially 1m / tick / MBP-10).
- **Canonical “tape” format** we can replay into multiple downstream shapes:
  - Parquet for backtests.
  - BigQuery `raw.*` tables for the ETL / training pipeline.
  - Local experiments without re-hitting the Databento API.
- **Clear separation of concerns**:
  - Databento client → DBN files (raw tape).
  - Our loaders → decode DBN → normalized tables (CBI schema).

## 2. Storage Layout (DBN as Raw Tape)

Target layout (once implemented):

- **Local (Mac / dev)**: `~/Databento/dbn/<venue>/<schema>/<symbol>/<YYYY>/<YYYY-MM-DD>.dbn.zst`
- **GCS (prod)**: `gs://cbi-v15-dbn/<venue>/<schema>/<symbol>/<YYYY>/<YYYY-MM-DD>.dbn.zst`

Examples:

- `gs://cbi-v15-dbn/glbx-mdp3/ohlcv-1m/ZL/2020/2020-01-02.dbn.zst`
- `gs://cbi-v15-dbn/glbx-mdp3/mbp-10/ZL/2020/2020-01-02.dbn.zst`

**Rule**: DBN files are immutable. If we need to fix anything, we re-download into a new location/cut, we don’t mutate existing tape.

## 3. Download Pattern (Historical DBN)

High-level pattern for a future loader (pseudocode, not yet implemented in `src/ingestion/databento`):

```python
from datetime import date
import databento as db

client = db.Historical(api_key=...)  # use keychain/secret manager

def download_dbn_ohlcv_1m(symbol: str, start: date, end: date):
    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="ohlcv-1m",
        symbols=[symbol],
        start=start,
        end=end,
        encoding="dbn",   # critical: DBN, not CSV
    )
    # store.save(path) or stream to GCS
```

Later extensions:

- Use Databento batch jobs for very long ranges.
- Parameterize `schema` (`ohlcv-1d`, `ohlcv-1m`, `mbp-10`, etc.) and symbol lists (`ZL`, `ZS`, `ZM`, FX crosses).

## 4. Decoding and Schema Mapping

Decoding DBN is conceptually:

1. **Load DBN** (local or via GCS stream).
2. **Decode to a typed frame** using Databento’s Python helpers.
3. **Map fields into our canonical BigQuery schema** for the corresponding `raw.*` table.

Target mapping for intraday OHLCV (example):

- DBN `ts` / `ts_event` → `timestamp` (TIMESTAMP, UTC) or split into `date` + `time`.
- DBN price fields → `open`, `high`, `low`, `close` (FLOAT64).
- DBN `volume` → `volume` (INT64).
- DBN `oi` / `open_interest` → `open_interest` (INT64) where available.
- Symbol → normalized `symbol` (STRING), consistent with current `raw.databento_futures_ohlcv_1d`.

Resulting BigQuery targets (examples):

- `raw.databento_futures_ohlcv_1m`
- `raw.databento_mbp10_ticks`

These are then consumed by new staging models (e.g. `staging.market_intraday`) and Python feature builders without changing the existing daily path.

## 5. Integration with CBI-V15 Pipeline

Once DBN ingestion is turned on, the flow will look like:

1. **Download DBN (historical or scheduled)**  
   - New script in `src/ingestion/databento/` (e.g. `download_dbn_intraday.py`).  
   - Writes DBN files to local disk and/or `gs://cbi-v15-dbn/...`.

2. **Decode DBN → BigQuery RAW**  
   - New script (e.g. `decode_dbn_to_raw.py`) reads DBN, decodes to DataFrame, writes to:
     - `raw.databento_futures_ohlcv_1m`
     - `raw.databento_mbp10_ticks` (later)

3. **STAGING → FEATURES → TRAINING (Python + SQL)**  
   - New SQL views and/or Python builders add staging models for intraday/microstructure.  
   - Existing `staging.market_daily`, `features.*`, and `training.*` remain intact.

4. **Mac training and forecasts**  
   - Training tables can optionally include intraday features derived from the new staging tables.  
   - Table contracts (column names/types) remain stable so additional features are additive.

## 6. Implementation Checklist (When We Turn This On)

- [ ] Create GCS bucket `cbi-v15-dbn` (or confirm existing).  
- [ ] Implement `download_dbn_intraday.py` with DBN encoding, path layout, and retries.  
- [ ] Implement `decode_dbn_to_raw.py` that:
  - Reads DBN from local/GCS.
  - Decodes to Pandas/Polars.
  - Writes to the appropriate `raw.*` tables with strict schema.  
- [ ] Add staging models for intraday/microstructure, wired into future Python feature tables.  
- [ ] Extend training tables to optionally consume intraday-derived features (keeping existing contracts stable).

Until those boxes are checked, DBN remains a future enhancement. This doc is here so we don’t have to rethink the architecture when we’re ready to pull the trigger.
