# Data Ingestion

Python scripts for collecting data from various sources. **Direct API calls** - NO Trigger.dev.

## Structure

```
src/ingestion/
├── fred/              # FRED Economic Data (Federal Reserve)
│   ├── collect_fred_fx.py
│   ├── collect_fred_financial_conditions.py
│   ├── collect_fred_rates_curve.py
│   └── collect_fred_priority_series.py
├── eia_epa/           # EIA/EPA Energy Data
│   ├── collect_eia_biofuels.py
│   └── collect_epa_rin_prices.py
├── databento/         # Futures Market Data
│   └── collect_daily.py
├── cftc/              # CFTC Commitments of Traders
│   └── ingest_cot.py
├── usda/              # USDA Agricultural Data
│   ├── ingest_export_sales.py
│   ├── ingest_wasde.py
│   └── collect_china_imports.py
├── scrapecreators/    # News & Sentiment Data
│   ├── collect_scrapecreators_news.py
│   ├── collect_by_big8_buckets.py
│   ├── collect_news_buckets.py
│   ├── collect_trump_social_media.py
│   └── sentiment_calculator.py
└── weather/           # NOAA Weather Data
    ├── collect_us_cornbelt.py
    ├── collect_brazil_soy_belt.py
    ├── collect_argentina_pampas.py
    └── collect_all_weather.py
```

## Usage

### Individual Collectors

```bash
# FRED data
python src/ingestion/fred/collect_fred_priority_series.py

# EIA/EPA data
python src/ingestion/eia_epa/collect_epa_rin_prices.py

# Databento futures
python src/ingestion/databento/collect_daily.py

# CFTC COT reports
python src/ingestion/cftc/ingest_cot.py

# USDA exports
python src/ingestion/usda/ingest_export_sales.py

# ScrapeCreators news
python src/ingestion/scrapecreators/collect_by_big8_buckets.py

# Weather data
python src/ingestion/weather/collect_all_weather.py
```

### Orchestration

See `src/orchestration/` for daily collection scripts:
- `collect_all_buckets.py` - Run all Big 8 bucket collectors

## Data Flow

```
Ingestion Scripts → data/raw/ → MotherDuck → data/duckdb/ → Training
```

## Environment Variables

Required in `.env`:
```
# FRED
FRED_API_KEY=...

# EIA
EIA_API_KEY=...

# Databento
DATABENTO_API_KEY=...

# MotherDuck
MOTHERDUCK_TOKEN=...
MOTHERDUCK_DB=cbi_v15

# ScrapeCreators (optional)
SCRAPECREATORS_API_KEY=...

# NOAA Weather
NOAA_API_TOKEN=...
```

## Legacy Code

Historical Trigger.dev files are archived in `archive/trigger/`:
- `.ts` files (TypeScript/Trigger.dev)
- Not actively maintained
- Kept for reference only