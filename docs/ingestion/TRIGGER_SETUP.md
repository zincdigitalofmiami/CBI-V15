# Trigger.dev Setup Guide for CBI-V15

**Purpose:** Orchestrate all ETL and training jobs with scheduling, logging, and retries.

**Based on:** [Trigger.dev Multi-Source ETL Pipeline](https://trigger.dev/docs/guides/use-cases/data-processing-etl#multi-source-etl-pipeline)

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

### Vercel Environment Variables

Add these in Vercel dashboard → Settings → Environment Variables:

```bash
# MotherDuck
MOTHERDUCK_TOKEN=your_token_here
MOTHERDUCK_DB=cbi_v15

# Data Sources
DATABENTO_API_KEY=your_key_here
FRED_API_KEY=your_key_here
EIA_API_KEY=your_key_here
SCRAPECREATORS_API_KEY=B1TOgQvMVSV6TDglqB8lJ2cirqi2

# Premium News Sources
PROFARMER_USERNAME=your_username
PROFARMER_PASSWORD=your_password
DTN_USERNAME=your_username
DTN_PASSWORD=your_password
JACOBSEN_USERNAME=your_username
JACOBSEN_PASSWORD=your_password

# Trigger.dev
TRIGGER_API_KEY=your_trigger_api_key
TRIGGER_API_URL=https://api.trigger.dev
```

### Local Development (.env)

```bash
# Copy from .env.example
cp .env.example .env

# Add Trigger.dev key
echo "TRIGGER_API_KEY=your_trigger_api_key" >> .env
```

---

## 4. Project Structure

```
CBI-V15/
├── trigger/                          # Trigger.dev jobs
│   ├── fred_seed_harvest.ts         # FRED series discovery
│   ├── fred_ingest_job.ts           # FRED daily observations
│   ├── profarmer_ingest_job.ts      # ProFarmer news scraping
│   ├── databento_ingest_job.ts      # Databento futures data
│   ├── eia_ingest_job.ts            # EIA biofuels data
│   ├── scrapecreators_ingest_job.ts # ScrapeCreators news
│   ├── news_to_signals_job.ts       # News sentiment processing
│   └── multi_source_etl.ts          # Master orchestrator
│
├── src/
│   ├── shared/
│   │   ├── motherduck_client.ts     # MotherDuck connection
│   │   ├── http_client.ts           # HTTP utilities
│   │   └── anchor_client.ts         # Anchor browser automation
│   │
│   └── ingestion/
│       └── buckets/                  # Bucket-level scrapers
│
├── trigger.config.ts                 # Trigger.dev configuration
└── package.json                      # Node dependencies
```

---

## 5. Implemented Jobs

### ✅ FRED Seed Harvest (`fred_seed_harvest.ts`)

**Purpose:** Discover FRED economic series using search API

**Search Categories:**
- **FX:** trade weighted, DEX, DXY, BRL, ARS, CNY, MXN
- **Rates:** treasury constant maturity, federal funds, sofr, term premium
- **Macro:** GDP, CPI, PCE, employment, industrial production, M2
- **Credit:** corporate bond spread, high yield
- **Financial Conditions:** stress index, risk appetite, volatility index, NFCI, STLFSI

**Output:** `raw.fred_series_metadata` table

**Trigger:**
```bash
npx trigger.dev@latest dev  # Start dev server
# Then trigger manually or via schedule
```

### ✅ ProFarmer Daily Ingest (`profarmer_ingest_job.ts`)

**Purpose:** Scrape ProFarmer daily editions

**Editions:**
- First Thing Today (pre_open)
- Ahead of the Open (pre_open)
- After the Bell (post_close)
- Agriculture News (intraday)
- Newsletters (newsletter)

**Output:** `raw.bucket_news` table

**Schedule:** 6 AM, 12 PM, 6 PM UTC (covers all editions)

### ✅ Multi-Source ETL (`multi_source_etl.ts`)

**Purpose:** Orchestrate all ingestion jobs

**Phase 1 (Parallel):**
- FRED seed harvest
- ProFarmer news
- Databento futures
- EIA biofuels

**Phase 2 (Sequential):**
- News-to-signals processing
- Feature engineering

**Schedule:** Daily at 7 AM UTC (2 AM ET)

---

## 6. Running Jobs

### Development Mode

```bash
# Start Trigger.dev dev server
npx trigger.dev@latest dev

# In another terminal, trigger jobs manually
npx trigger.dev@latest trigger fred-seed-harvest
npx trigger.dev@latest trigger profarmer-daily-ingest
npx trigger.dev@latest trigger multi-source-etl
```

### Production Deployment

```bash
# Deploy to Trigger.dev cloud
npx trigger.dev@latest deploy

# Jobs will run on schedule automatically
```

---

## 7. Monitoring

### Trigger.dev Dashboard

- View job runs: https://cloud.trigger.dev/projects/cbi-v15/runs
- Check logs and errors
- Retry failed jobs
- Adjust schedules

### MotherDuck Verification

```sql
-- Check FRED series discovered
SELECT category, COUNT(*) as series_count
FROM raw.fred_series_metadata
GROUP BY category;

-- Check ProFarmer articles
SELECT edition_type, COUNT(*) as article_count
FROM raw.bucket_news
WHERE source = 'ProFarmer'
GROUP BY edition_type;

-- Check ETL pipeline status
SELECT source, success, records_processed, timestamp
FROM logs.etl_runs
ORDER BY timestamp DESC
LIMIT 10;
```

---

## 8. Next Steps

**TODO - Remaining Jobs:**
- [ ] `databento_ingest_job.ts` - Futures prices
- [ ] `eia_ingest_job.ts` - Biofuels data
- [ ] `scrapecreators_ingest_job.ts` - News buckets
- [ ] `news_to_signals_job.ts` - Sentiment processing
- [ ] `macro_series_discovery_job.ts` - Auto-discover new FRED series

**TODO - Enhancements:**
- [ ] Error alerting (Slack/email)
- [ ] Data quality checks
- [ ] Backfill jobs for historical data
- [ ] Rate limiting for API calls

---

**Last Updated:** December 9, 2024

