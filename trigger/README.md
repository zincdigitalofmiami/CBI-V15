# Trigger.dev Jobs - CBI-V15 Orchestration

> Fast-moving workspace: always check the latest `TRIGGER_SETUP.md`, `WEB_SCRAPING_TARGETS_MASTER.md`, `DATA_LINKS_MASTER.md`, and active master plan `.cursor/plans/ALL_PHASES_INDEX.md` before editing or adding jobs. Keep explorer clean; no duplicate jobs/scripts/MDs.

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
├── Adapters/                      # Shared utilities (future)
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
├── DirectScrapers/                # Direct website scraping (not API)
│   ├── Guides/
│   ├── Scripts/
│   │   ├── collect_china_news.py
│   │   └── collect_tariff_news.py
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
├── Orchestration/                 # Cross-source orchestration
│   ├── collect_all_buckets.py
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
│   │   ├── buckets/
│   │   │   ├── collect_biofuel_policy.py
│   │   │   ├── collect_china_demand.py
│   │   │   ├── collect_tariffs_trade_policy.py
│   │   │   └── collect_trump_truth_social.py
│   │   ├── collect_news_buckets.py
│   │   ├── direct_url_scraper.py
│   │   ├── sentiment_calculator.py
│   │   ├── intelligent_news_pipeline.ts
│   │   └── news_to_signals_openai_agent.ts
│   └── README.md
│
├── TradingEconomics/              # Global commodity data
│   ├── Guides/
│   ├── Scripts/
│   │   ├── tradingeconomics_anchor.py
│   │   └── tradingeconomics_goldmine.ts
│   └── README.md
│
├── UofI_Feeds/                    # University of Illinois (CRITICAL)
│   ├── Guides/
│   ├── Scripts/                   # ⚠️ NEEDS IMPLEMENTATION
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
- ✅ Ingestion lives in `trigger/<Source>/Scripts/` (canonical)
- ⚠️ `src/ingestion/` is legacy pointers only; no active ingestion logic

---

## Status Snapshot (Dec 2025)

### ✅ Implemented
- DataBento: `collect_daily.py`, `databento_ingest_job.ts`
- FRED: `collect_fred_*.py`, `fred_seed_harvest.ts`
- EIA_EPA: `collect_eia_biofuels.py`, `eia_procurement_ingest.ts`
- ScrapeCreators: `collect_news_buckets.py`, `intelligent_news_pipeline.ts`, bucket collectors
- ProFarmer: `profarmer_*.ts`, `profarmer_anchor.py`
- Vegas: `collect_vegas_intel.py`, `vegas_intel_job.ts`
- CFTC: `ingest_cot.py`
- USDA: `ingest_export_sales.py`, `ingest_wasde.py`
- Weather: `ingest_weather.py`
- DirectScrapers: `collect_china_news.py`, `collect_tariff_news.py`

### ⚠️ Needs Implementation
- EPA RIN prices: `EIA_EPA/Scripts/epa_rin_prices.ts`
- Farm Policy News: `UofI_Feeds/Scripts/farmpolicynews.ts` (CRITICAL)
- farmdoc Daily: `UofI_Feeds/Scripts/farmdoc_daily.ts` (CRITICAL)
- CFTC COT Trigger job: `CFTC/Scripts/cftc_cot_reports.ts`
- USDA FAS Trigger job: `USDA/Scripts/usda_fas_exports.ts`

---

## Configuration

### Environment (common)
- `MOTHERDUCK_TOKEN`, `MOTHERDUCK_DB`
- `DATABENTO_API_KEY`, `FRED_API_KEY`, `EIA_API_KEY`
- `SCRAPECREATORS_API_KEY`
- `PROFARMER_USERNAME`, `PROFARMER_PASSWORD`, `ANCHOR_API_KEY`
- `OPENAI_API_KEY` (for AI-powered jobs)

### Scheduling (typical)
- Intraday: Market data, news (15–60 min)
- Daily: Macro, policy, weather (1–3x/day)
- Weekly: USDA exports, CFTC COT
- Monthly: WASDE, production reports

---

## Development (local)
```bash
cd trigger
npm install
# run individual job locally
npx trigger.dev run path/to/job.ts
# deploy (when connected)
npx trigger.dev deploy
```

**Last Updated:** December 13, 2025
