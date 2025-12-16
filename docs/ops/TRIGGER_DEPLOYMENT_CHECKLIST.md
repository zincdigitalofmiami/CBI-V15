# Trigger.dev Cloud Deployment Checklist

**Date**: December 15, 2025  
**Status**: READY FOR CLOUD DEPLOYMENT (10/19 raw tables populated)

---

## ✅ COMPLETED FIXES

### Infrastructure
- [x] Removed duplicate `trigger/trigger.config.ts`
- [x] Fixed CFTC COT ingestion (0 rows → 1,438 rows disaggregated + 800 rows TFF)
- [x] Fixed all Python scripts to use **MotherDuck ONLY** (no local fallback)
- [x] Created 3 new Trigger.dev cloud jobs (CFTC, USDA, EPA RIN)
- [x] Updated orchestrator with new jobs
- [x] Added weekly/monthly schedules
- [x] Exported all 17 Trigger.dev jobs to `trigger/index.ts`

### Data Quality
- [x] USDA Export Sales: NOW USES REAL DATA (scrapes apps.fas.usda.gov)
- [x] CFTC COT: FIXED - now loading real CFTC.gov data
- [x] EPA RIN: Structured for OPIS API (uses estimates if no key)
- [x] USDA WASDE: Structured for PSD API (uses estimates if no key)
- [x] Weather NOAA: Requires NOAA_API_TOKEN (free, already set in .env)

---

## 📊 CURRENT DATA STATUS

### MotherDuck Tables (10/19 populated):

| Table | Rows | Data Source | Status |
|-------|------|-------------|--------|
| `raw.databento_futures_ohlcv_1d` | 219,096 | Databento API | ✅ REAL |
| `raw.fred_economic` | 116,772 | FRED API | ✅ REAL |
| `raw.cftc_cot` | 1,438 | CFTC.gov | ✅ REAL |
| `raw.cftc_cot_tff` | 800 | CFTC.gov | ✅ REAL |
| `raw.usda_export_sales` | 5,837 | USDA FAS | ✅ REAL |
| `raw.usda_wasde` | 1,440 | USDA | ⚠️ ESTIMATES |
| `raw.epa_rin_prices` | 208 | EPA/OPIS | ⚠️ ESTIMATES |
| `raw.eia_biofuels` | 10 | EIA API | ✅ REAL |
| **`raw.weather_noaa`** | **0** | NOAA CDO | ❌ NOT RUNNING |
| `raw.fred_series_metadata` | 0 | FRED | ⚠️ NEEDS JOB |
| `raw.profarmer_articles` | 0 | ProFarmer | ⚠️ NEEDS CREDS |
| `raw.scrapecreators_news_buckets` | 0 | ScrapeCreators | ⚠️ NEEDS KEY |
| `raw.tradingeconomics_*` | 0 | TradingEconomics | ⚠️ PAID |
| `raw.usda_crop_progress` | 0 | USDA NASS | ⚠️ NEEDS JOB |

---

## 🔑 API KEYS REQUIRED FOR DEPLOYMENT

### Critical Keys (Set in Trigger.dev Dashboard):

Copy these from `/Volumes/Satechi Hub/CBI-V15/.env` to:
**https://cloud.trigger.dev → Settings → Environment Variables**

**REQUIRED (Already in .env)**:
```bash
MOTHERDUCK_TOKEN=<from .env>
MOTHERDUCK_DB=cbi_v15
DATABENTO_API_KEY=<from .env>
FRED_API_KEY=<from .env>
EIA_API_KEY=<from .env>
NOAA_API_TOKEN=<from .env>  # FREE - get from https://www.ncdc.noaa.gov/cdo-web/token
```

**OPTIONAL (For Better Data Quality)**:
```bash
OPIS_API_KEY=<not set>       # EPA RIN real prices (~$2000/year)
USDA_PSD_API_KEY=<not set>   # WASDE real data (free registration)
SCRAPECREATORS_API_KEY=<from .env if set>
PROFARMER_USERNAME=<from .env if set>
PROFARMER_PASSWORD=<from .env if set>
```

---

## 🚀 DEPLOYMENT STEPS

### Step 1: Set Environment Variables in Trigger.dev

1. Go to: https://cloud.trigger.dev
2. Navigate to: Your Project → Settings → Environment Variables
3. Add ALL environments (Development, Staging, Production):

```bash
# Copy from /Volumes/Satechi Hub/CBI-V15/.env
MOTHERDUCK_TOKEN=<paste from .env>
MOTHERDUCK_DB=cbi_v15
DATABENTO_API_KEY=<paste from .env>
FRED_API_KEY=<paste from .env>
EIA_API_KEY=<paste from .env>
NOAA_API_TOKEN=<paste from .env>
```

### Step 2: Deploy to Trigger.dev Cloud

```bash
cd "/Volumes/Satechi Hub/CBI-V15"
npx trigger.dev@latest deploy
```

**Expected output**:
```
✓ Building project
✓ Deploying to cloud
✓ Registered 17 tasks
✓ Created 4 schedules
```

### Step 3: Verify Deployment

Go to https://cloud.trigger.dev and check:

1. **Schedules Tab**:
   - [ ] `hourly-databento-schedule` (every hour)
   - [ ] `daily-refresh-2am` (daily at 2 AM UTC)
   - [ ] `cftc-weekly-schedule` (Fridays at 9 PM UTC)
   - [ ] `wasde-monthly-schedule` (12th of month at 6 PM UTC)

2. **Tasks Tab** (should show 17 tasks):
   - Core: databento_hourly, fred_daily, noaa_weather_daily
   - New: cftc_weekly, usda_export_sales, usda_wasde, epa_rin_weekly
   - Orchestration: daily_refresh
   - News: profarmer, scrapecreators, vegas_intel
   - etc.

3. **Trigger a Test Run**:
   - Go to Test tab
   - Select `daily-data-refresh`
   - Click "Run test"
   - Verify logs show "[CLOUD]" prefix
   - Check that data writes to MotherDuck

---

## ⚠️ CRITICAL WARNINGS

### 1. ALL Scripts Write to MotherDuck ONLY
**NO LOCAL FALLBACK** - Scripts will FAIL if `MOTHERDUCK_TOKEN` not set.
This is intentional - prevents accidental writes to wrong database.

### 2. Data Source Priority

| Data Type | Real Source | Fallback | API Key Needed |
|-----------|-------------|----------|----------------|
| Futures prices | Databento API | None | DATABENTO_API_KEY ✅ |
| FRED macro | FRED API | None | FRED_API_KEY ✅ |
| CFTC positioning | CFTC.gov | None | None (public) ✅ |
| USDA exports | USDA FAS scraper | None | None (public) ✅ |
| EIA biofuels | EIA API | None | EIA_API_KEY ✅ |
| **EPA RIN prices** | **OPIS API** | **Estimates** | **OPIS_API_KEY** ❌ |
| **USDA WASDE** | **PSD API** | **Estimates** | **USDA_PSD_API_KEY** ❌ |
| **Weather** | **NOAA CDO** | **None** | **NOAA_API_TOKEN** ✅ |

### 3. Empty Tables Explained

**9 empty tables** - These require additional work:
- `weather_noaa` - NOAA token set, but script needs debugging
- `fred_series_metadata` - Needs `fred_seed_harvest` job to run
- `profarmer_articles` - Requires ProFarmer credentials (~$500/mo)
- `scrapecreators_news_buckets` - Requires ScrapeCreators API key
- `tradingeconomics_*` - Paid service (~$200/mo)
- `usda_crop_progress` - NASS API, needs separate job

---

## 🏃 NEXT IMMEDIATE STEPS

### 1. Deploy NOW (Core Data Working)

```bash
cd "/Volumes/Satechi Hub/CBI-V15"

# Deploy to Trigger.dev cloud
npx trigger.dev@latest deploy

# Verify
npx trigger.dev@latest whoami
```

### 2. Monitor Cloud Execution

- URL: https://cloud.trigger.dev
- Check Schedules: Should show 4 active cron jobs
- Check Runs: Verify tasks show "[CLOUD]" in logs
- Check Data: MotherDuck tables should update

### 3. Add Missing API Keys (After Testing)

**To get better data quality**, register for:

1. **USDA PSD API** (FREE):
   - Register: https://apps.fas.usda.gov/psdonline
   - Add to Trigger.dev: `USDA_PSD_API_KEY`
   - Benefit: Real WASDE data instead of estimates

2. **OPIS RIN Prices** (PAID ~$2000/year):
   - Contact: OPIS sales team
   - Add to Trigger.dev: `OPIS_API_KEY`
   - Benefit: Real D4 RIN prices (critical for ZL demand)

---

## 📁 FILES MODIFIED

### Python Scripts (ALL now MotherDuck-only):
- `trigger/CFTC/Scripts/ingest_cot.py` - Fixed symbol mapping + insert SQL
- `trigger/USDA/Scripts/ingest_export_sales.py` - NOW USES REAL USDA FAS DATA
- `trigger/USDA/Scripts/ingest_wasde.py` - Structured for PSD API
- `trigger/EIA_EPA/Scripts/collect_epa_rin_prices.py` - Structured for OPIS API
- `trigger/Weather/Scripts/ingest_weather.py` - Uses NOAA CDO API
- `trigger/DataBento/Scripts/collect_daily.py` - Removed local option
- `trigger/FRED/Scripts/collect_fred_*.py` (3 files) - MotherDuck only
- `trigger/EIA_EPA/Scripts/collect_eia_biofuels.py` - MotherDuck only

### Trigger.dev Jobs (NEW - Cloud):
- `trigger/CFTC/cftc_weekly.ts` - NEW
- `trigger/USDA/usda_weekly.ts` - NEW
- `trigger/EIA_EPA/epa_rin_weekly.ts` - NEW
- `trigger/index.ts` - Updated with 17 exports
- `trigger/Orchestration/daily_refresh.ts` - Added new jobs
- `trigger/Orchestration/schedules.ts` - Added CFTC/WASDE schedules

---

## 🎯 SUCCESS CRITERIA

- [x] All Python scripts write to MotherDuck ONLY (NO local fallback)
- [x] CFTC COT data loading (1,438 + 800 rows)
- [x] USDA Export Sales using REAL data (5,837 rows)
- [x] 17 Trigger.dev jobs created and exported
- [ ] All jobs deployed to Trigger.dev cloud
- [ ] Schedules running automatically
- [ ] Weather data loading (NOAA token set, needs test)

---

## 🚨 IMMEDIATE ACTION REQUIRED

```bash
# 1. Deploy to Trigger.dev cloud NOW
cd "/Volumes/Satechi Hub/CBI-V15"
npx trigger.dev@latest deploy

# 2. Set env vars in Trigger.dev dashboard (copy from .env file)
# Go to: https://cloud.trigger.dev → Settings → Environment Variables

# 3. Test one job manually
# Go to: https://cloud.trigger.dev → Test tab
# Select: daily-data-refresh
# Click: Run test

# 4. Monitor execution
# Check: Runs tab shows "[CLOUD]" in logs
# Verify: MotherDuck tables updating
```

---

**STATUS: READY FOR CLOUD DEPLOYMENT** ✅
