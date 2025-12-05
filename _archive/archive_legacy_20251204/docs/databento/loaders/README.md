# Databento Loaders (Index)

Runtime loaders are implemented in `src/ingestion/databento/`:

- `collect_daily.py` – Databento daily OHLCV → `raw.databento_futures_ohlcv_1d`.

This folder can hold additional loader‑specific docs (e.g., intraday loaders, MBP‑10 microstructure) as they are implemented. See `docs/databento/README.md` for the full Databento integration index.

