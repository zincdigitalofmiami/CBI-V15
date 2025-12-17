# Data Directory

Local data storage for CBI-V15. **Not tracked in git** (large files).

## Structure

```
data/
├── duckdb/              # Local DuckDB mirror (synced from MotherDuck)
│   └── cbi_v15.duckdb   # Training data landing pad
├── models/              # AutoGluon model artifacts
│   └── zl_forecast/     # ZL forecasting model
├── raw/                 # Raw data landing (from ingestion scripts)
│   ├── fred/            # FRED economic data (parquet)
│   ├── eia/             # EIA biofuels data (parquet)
│   ├── databento/       # Databento futures OHLCV
│   ├── weather/         # NOAA weather data
│   ├── cftc/            # CFTC COT reports
│   ├── usda/            # USDA export sales, WASDE
│   └── scrapecreators/  # News/sentiment data
└── README.md
```

## Data Flow

```
Ingestion Scripts (trigger/)
         │
         ▼
   data/raw/*.parquet     ← Raw data landing
         │
         ▼
   MotherDuck (cloud)     ← Source of truth
         │
         ▼
   data/duckdb/           ← Local mirror for training
         │
         ▼
   AutoGluon Training
         │
         ▼
   data/models/           ← Model artifacts
```

## Usage

### Sync MotherDuck to Local
```bash
python scripts/sync_motherduck_to_local.py
```

### Train Models (reads from local DuckDB)
```bash
python src/training/autogluon/timeseries_trainer.py
```

### Export Models
Models saved to `data/models/` are uploaded to MotherDuck for dashboard consumption.

## Git Ignore

All data files are git-ignored:
- `*.duckdb`, `*.db` - Database files
- `*.parquet`, `*.csv` - Data files
- `*.pkl`, `*.joblib` - Model artifacts

Only `.gitignore` and `README.md` are tracked.

## Archive

Historical/backup data is in `archive/Data/`:
- `archive/Data/databento/` - Historical Databento downloads (866MB)
- `archive/Data/raw/` - Old raw data files


