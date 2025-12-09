# Ingestion (Data Collectors)

## Purpose
Data collectors - one subfolder per data source. Each collector fetches data and writes to MotherDuck `raw.*` tables.

## Structure
```
ingestion/
├── databento/       # Market data (ZL, ZS, ZC futures)
├── eia/             # EIA biofuels data
├── fred/            # FRED macro data (FX, rates, financial conditions)
├── scrape_creator/  # News buckets (Trump, tariffs, biofuel policy)
├── legacy_weather/  # Weather data (to be modernized)
└── [future]/
    ├── cftc/        # CFTC COT data
    ├── ers/         # USDA ERS data
    ├── gdelt/       # GDELT events
    └── news/        # ProFarmer, TradingEconomics
```

## What Belongs Here
- `collect_*.py` - Data fetching scripts
- Source-specific utilities
- Each subfolder should have `__init__.py` and `README.md`

## What Does NOT Belong Here
- Ingestion configuration (→ `config/ingestion/`)
- Operational scripts (→ `scripts/ops/`)
- Data transformations (→ `database/definitions/02_staging/`)

## Naming Convention
- Folder: `{source_name}/` (lowercase, underscore)
- Main collector: `collect_{data_type}.py`
- Example: `fred/collect_fred_fx.py`

## Output
All collectors write to MotherDuck `raw.*` tables as defined in `database/definitions/01_raw/`
