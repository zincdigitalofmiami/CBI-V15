# Trigger.dev Jobs - CBI-V15 Orchestration

**Purpose:** Domain-organized ingestion and orchestration for CBI-V15.

**Key Principle:** Ingestion is organized by **source**, not topic. Each source has `Guides/`, `Scripts/`, and `README.md`.

---

## Directory Structure

```
trigger/
├── trigger.config.ts              # Main Trigger.dev configuration
├── multi_source_etl.ts            # Master ETL orchestrator
├── README.md                      # This file
├── TRIGGER_SETUP.md               # Setup guide
├── WEB_SCRAPING_TARGETS_MASTER.md # ALL URLs to scrape
│
├── Adapters/                      # Shared utilities
│   └── README.md
│
├── Analysts/                      # Multi-source analyst/news scrapers
│   ├── Guides/
│   ├── Scripts/
│   │   └── collect_china_news.py
│   └── README.md
│
├── CFTC/                          # Commitment of Traders
│   ├── Guides/
│   │   └── CFTC_COT_INGESTION.md
│   ├── Scripts/
│   │   └── ingest_cot.py
│   └── README.md
│
├── DataBento/                     # Market data
│   ├── Guides/
│   ├── Scripts/
│   │   ├── collect_daily.py
│   │   └── databento_ingest_job.ts
│   └── README.md
│
├── EIA_EPA/                       # Energy & biofuels
│   ├── Guides/
│   ├── Scripts/
│   │   ├── collect_eia_biofuels.py
│   │   └── eia_procurement_ingest.ts
│   └── README.md
│
├── FRED/                          # Economic data
│   ├── Guides/
│   ├── Scripts/
│   │   ├── collect_fred_fx.py
│   │   ├── collect_fred_financial_conditions.py
│   │   ├── collect_fred_rates_curve.py
│   │   └── fred_seed_harvest.ts
│   └── README.md
│
├── Policy/                        # Policy/trade scrapers
│   ├── Guides/
│   ├── Scripts/
│   │   └── collect_tariff_news.py
│   └── README.md
│
├── ProFarmer/                     # Premium ag intelligence
│   ├── Guides/
│   ├── Scripts/
│   │   ├── profarmer_all_urls.ts
│   │   ├── profarmer_anchor.py
│   │   ├── profarmer_anchor_scraper.ts
│   │   └── profarmer_ingest_job.ts
│   └── README.md
│
├── ScrapeCreators/                # News & sentiment API
│   ├── Guides/
│   │   └── NEWS_PIPELINE.md
│   ├── Scripts/
│   │   ├── buckets/*.py
│   │   ├── collect_news_buckets.py
│   │   ├── direct_url_scraper.py
│   │   ├── sentiment_calculator.py
│   │   ├── intelligent_news_pipeline.ts
│   │   └── news_to_signals_openai_agent.ts
│   └── README.md
│
├── Scripts/                       # Cross-source orchestration
│   └── collect_all_buckets.py
│
├── TradingEconomics/              # Global commodity data
│   ├── Guides/
│   ├── Scripts/
│   │   ├── tradingeconomics_anchor.py
│   │   └── tradingeconomics_goldmine.ts
│   └── README.md
│
├── UofI_Feeds/                    # University of Illinois
│   ├── Guides/
│   ├── Scripts/
│   └── README.md
│
├── USDA/                          # Agricultural data
│   ├── Guides/
│   ├── Scripts/
│   │   ├── ingest_export_sales.py
│   │   └── ingest_wasde.py
│   └── README.md
│
├── Vegas/                         # Vegas events intel
│   ├── Guides/
│   ├── Scripts/
│   │   ├── collect_vegas_intel.py
│   │   └── vegas_intel_job.ts
│   └── README.md
│
└── Weather/                       # Weather & climate
    ├── Guides/
    ├── Scripts/
    │   └── ingest_weather.py
    └── README.md
```

---

## Source-Based Organization

- `Guides/` - Markdown documentation, walkthroughs, operator notes
- `Scripts/` - Trigger.dev jobs (TypeScript) and ingestion runners (Python)
- `README.md` - 3-5 line description of the source and what the scripts do

---

## Single Source of Truth

- ✅ **Canonical ingestion** → `trigger/<Source>/Scripts/`
- ⚠️ `src/ingestion/` now only holds legacy pointers/README stubs; no active ingestion logic

---

## Active Jobs

| Job | Location | Target | Status |
|-----|----------|--------|--------|
| `collect_daily.py` | `DataBento/Scripts/` | `raw.databento_ohlcv_daily` | ✅ Active |
| `collect_fred_rates_curve.py` | `FRED/Scripts/` | `raw.fred_economic` | ✅ Active |
| `collect_eia_biofuels.py` | `EIA_EPA/Scripts/` | `raw.eia_biofuels` | ✅ Active |
| `collect_news_buckets.py` | `ScrapeCreators/Scripts/` | `raw.scrapecreators_news_buckets` | ✅ Active |
| `collect_all_buckets.py` | `Scripts/` | `raw.bucket_news` | ✅ Active |
| `profarmer_all_urls.ts` | `ProFarmer/Scripts/` | `raw.bucket_news` | ✅ Active |
| `ingest_cot.py` | `CFTC/Scripts/` | `raw.cftc_cot_*` | ✅ Active |
| `ingest_wasde.py` | `USDA/Scripts/` | WASDE exports | ✅ Active |

---

## Priority Implementation Order

### Phase 1: Critical Ingestion Jobs (NEEDED)
1. **EPA RIN Prices** - `EIA_EPA/Scripts/epa_rin_prices.ts`
2. **USDA Export Sales** - `USDA/Scripts/usda_fas_exports.ts`
3. **CFTC COT** - `CFTC/Scripts/cftc_cot_reports.ts`

### Phase 2: News & Intelligence (NEEDED)
4. **Farm Policy News** - `UofI_Feeds/Scripts/farmpolicynews.ts`
5. **farmdoc Daily** - `UofI_Feeds/Scripts/farmdoc_daily.ts`

### Phase 3: Weather (NEEDED)
6. **NOAA Weather** - `Weather/Scripts/noaa_weather.ts`
7. **INMET Brazil** - `Weather/Scripts/inmet_brazil.ts`

---

## Configuration

### Environment Variables
All jobs read from `process.env`:
- `TRIGGER_SECRET_KEY`
- `MOTHERDUCK_TOKEN`
- `MOTHERDUCK_DB`
- `DATABENTO_API_KEY`
- `FRED_API_KEY`
- `EIA_API_KEY`
- `SCRAPECREATORS_API_KEY`
- `PROFARMER_USERNAME`
- `PROFARMER_PASSWORD`
- `OPENAI_API_KEY`
- `ANCHOR_API_KEY`

### Scheduling
- **Intraday**: Market data, news (every 15min - 1hr)
- **Daily**: Macro data, weather, policy (1-3x daily)
- **Weekly**: USDA exports, CFTC COT (weekly after release)
- **Monthly**: WASDE, production reports (monthly after release)

---

## Development

### Setup
```bash
# Install Trigger.dev CLI
npm install -g @trigger.dev/cli

# Initialize project
cd trigger
npx trigger.dev init

# Deploy jobs
npx trigger.dev deploy
```

### Testing
```bash
# Test individual job
npx trigger.dev run {job-name}

# Monitor jobs
npx trigger.dev logs
```

---

**Last Updated:** December 10, 2025
