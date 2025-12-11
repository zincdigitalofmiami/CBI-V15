# Trigger.dev Setup Guide for CBI-V15

**Purpose:** Orchestrate all ETL and training jobs with scheduling, logging, and retries.

---

## 1. Create Trigger.dev Account

1. Go to [trigger.dev](https://trigger.dev)
2. Sign up with GitHub
3. Create new project: `cbi-v15`
4. Link to GitHub repo: `zincdigitalofmiami/CBI-V15`

---

## 2. Install Trigger.dev SDK

```bash
cd /Volumes/Satechi\ Hub/CBI-V15

# Install Trigger.dev CLI
npm install -g @trigger.dev/cli

# Initialize Trigger.dev in project
npx trigger.dev@latest init

# Install dependencies
npm install @trigger.dev/sdk duckdb
```

---

## 3. Configure Environment Variables

**All credentials stored in:**
- Local: `.env.local` (gitignored)
- Production: Vercel Environment Variables
- macOS: Apple Keychain (backup)

**Required variables:**
- `MOTHERDUCK_TOKEN`
- `MOTHERDUCK_DB`
- `DATABENTO_API_KEY`
- `FRED_API_KEY`
- `EIA_API_KEY`
- `SCRAPECREATORS_API_KEY`
- `PROFARMER_USERNAME`
- `PROFARMER_PASSWORD`
- `TRIGGER_SECRET_KEY`
- `OPENAI_API_KEY`
- `ANCHOR_API_KEY`

---

## 4. Project Structure

```
trigger/
├── trigger.config.ts              # Main Trigger.dev configuration
├── README.md                      # Jobs documentation
├── TRIGGER_SETUP.md               # This file
├── WEB_SCRAPING_TARGETS_MASTER.md # ALL URLs to scrape
│
├── Adapters/                      # Shared utilities
├── Analysts/                      # Analyst/news scrapers (e.g., collect_china_news.py)
├── CFTC/                          # COT ingestion (ingest_cot.py)
├── DataBento/                     # Market data (collect_daily.py, databento_ingest_job.ts)
├── EIA_EPA/                       # Energy & biofuels (collect_eia_biofuels.py, eia_procurement_ingest.ts)
├── FRED/                          # Economic data (collect_fred_*.py, fred_seed_harvest.ts)
├── Policy/                        # Policy/trade scrapers (collect_tariff_news.py)
├── ProFarmer/                     # Premium ag intel (profarmer_all_urls.ts, profarmer_anchor.py)
├── ScrapeCreators/                # ScrapeCreators API + helpers (collect_news_buckets.py)
├── Scripts/                       # Cross-source orchestration (collect_all_buckets.py)
├── TradingEconomics/              # TradingEconomics (tradingeconomics_anchor.py, tradingeconomics_goldmine.ts)
├── UofI_Feeds/                    # University of Illinois feeds
├── USDA/                          # Agricultural data (ingest_export_sales.py, ingest_wasde.py)
├── Vegas/                         # Vegas intel (collect_vegas_intel.py, vegas_intel_job.ts)
└── Weather/                       # Weather/NOAA (ingest_weather.py)
```

---

## 5. Running Jobs

### Development Mode

```bash
# Start Trigger.dev dev server
npx trigger.dev@latest dev

# Trigger jobs manually
npx trigger.dev@latest trigger profarmer-all-urls
```

### Production Deployment

```bash
# Deploy to Trigger.dev cloud
npx trigger.dev@latest deploy
```

---

## 6. Monitoring

### Trigger.dev Dashboard
- View job runs: https://cloud.trigger.dev/projects/cbi-v15/runs
- Check logs and errors
- Retry failed jobs
- Adjust schedules

---

## 7. Active Jobs

| Job | Location | Schedule | Status |
|-----|----------|----------|--------|
| ProFarmer All URLs | `ProFarmer/Scripts/profarmer_all_urls.ts` | 3x daily | ✅ Active |
| Databento Daily | `DataBento/Scripts/collect_daily.py` | Hourly / 4-hour | ✅ Active |
| FRED Rates Curve | `FRED/Scripts/collect_fred_rates_curve.py` | Daily 6 PM ET | ✅ Active |
| EIA Biofuels | `EIA_EPA/Scripts/collect_eia_biofuels.py` | Weekly | ✅ Active |
| ScrapeCreators Buckets | `ScrapeCreators/Scripts/collect_news_buckets.py` | Hourly | ✅ Active |
| CFTC COT | `CFTC/Scripts/ingest_cot.py` | Weekly | ✅ Active |

---

## 8. Jobs Needing Creation

See `trigger/WEB_SCRAPING_TARGETS_MASTER.md` for complete list.

**Critical Priority:**
1. EPA RIN Prices (`EIA_EPA/Scripts/epa_rin_prices.ts`)
2. Farm Policy News (`UofI_Feeds/Scripts/farmpolicynews.ts`)
3. farmdoc Daily (`UofI_Feeds/Scripts/farmdoc_daily.ts`)
4. USDA Export Sales (`USDA/Scripts/usda_fas_exports.ts`)
5. CFTC COT Reports (`CFTC/Scripts/cftc_cot_reports.ts`)

---

**Last Updated:** December 10, 2025
