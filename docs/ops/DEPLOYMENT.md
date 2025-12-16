# CBI-V15 Deployment Guide

## ✅ COMPLETED INFRASTRUCTURE

### Database Layer (MotherDuck)
- **77 tables** across 9 schemas (raw, staging, features, training, forecasts, reference, ops, explanations, main)
- **ALL Big 8 buckets preserved** (Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility)
- **Weather locations**: US Corn Belt, Brazil (Mato Grosso), Argentina (Pampas)

### Data Ingestion (Active)
- ✅ Databento: 219K rows (56 symbols, 2010-2025)
- ✅ FRED: 116K rows (24 series)
- ✅ EPA RIN: 208 rows (D3/D4/D5/D6 weekly)
- ✅ USDA Export Sales: 5.5K rows
- ✅ EIA Biofuels: 10 rows

### Trigger.dev Orchestration (Ready)
- **Hourly**: Databento updates
- **Daily**: Full refresh at 2 AM UTC (Databento, FRED, NOAA weather)
- **Coordinator pattern**: Parallel execution with real-time progress

### Dashboard (Live)
- Live API endpoint: `/api/live/databento`
- Auto-refresh: Every 5 minutes
- Charts: ForecastFanChart + Big8Panel

---

## 🚀 DEPLOYMENT STEPS (CLOUD EXECUTION ONLY)

### 1. Configure Environment Variables in Trigger.dev Dashboard

**CRITICAL**: All tasks run in Trigger.dev cloud, NOT on local machine.

Go to: https://cloud.trigger.dev → Your Project → Settings → Environment Variables

Add these for ALL environments (Development, Staging, Production):

```
MOTHERDUCK_TOKEN=your_motherduck_token
MOTHERDUCK_DB=cbi_v15
DATABENTO_API_KEY=your_databento_key
FRED_API_KEY=your_fred_key
EIA_API_KEY=your_eia_key
NOAA_API_TOKEN=your_noaa_token
```

**Reference**: [Trigger.dev Environment Variables](https://trigger.dev/docs/deployment/environment-variables)

### 2. Deploy Trigger.dev Jobs to Cloud

```bash
cd /Volumes/Satechi\ Hub/CBI-V15
npx trigger.dev@latest deploy
```

This deploys to **Trigger.dev cloud servers** (NOT your local machine):
- `databento-hourly-update` - Runs every hour in cloud
- `fred-daily-update` - Runs daily at 2 AM in cloud
- `noaa-weather-daily` - Runs daily at 2 AM in cloud
- `daily-data-refresh` - Coordinator, runs daily at 2 AM in cloud

**All tasks execute in Trigger.dev's infrastructure with:**
- Machine: small-1x (0.5 vCPU, 1 GB RAM)
- Max concurrency: 15 parallel tasks
- Queue: data-ingestion (prevents overload)

### 3. Verify Cloud Execution in Dashboard

Go to: https://cloud.trigger.dev

Check:
- ✅ Schedules tab shows active cron jobs
- ✅ Runs tab shows executions (in CLOUD, not local)
- ✅ Logs show "[CLOUD]" prefix
- ✅ No errors in recent runs

### 4. Deploy Dashboard (Vercel)

```bash
cd dashboard
vercel deploy --prod
```

Add environment variables in Vercel dashboard:
- `MOTHERDUCK_TOKEN`
- `MOTHERDUCK_DB=cbi_v15`
- `MOTHERDUCK_PROXY_URL=http://localhost:8000` (if using proxy)

### 4. Test Live Dashboard

Visit deployed URL and verify:
- ✅ ZL price chart loads with live data
- ✅ Latest price shows (from MotherDuck)
- ✅ Auto-refresh works (5 min intervals)
- ✅ No console errors

---

## 📊 MONITORING

### Check Data Freshness

```bash
python3 -c "
import duckdb, os
conn = duckdb.connect(f'md:cbi_v15?motherduck_token={os.getenv(\"MOTHERDUCK_TOKEN\")}')

# Check latest data dates
tables = ['databento_futures_ohlcv_1d', 'fred_economic', 'epa_rin_prices']
for table in tables:
    result = conn.execute(f'SELECT MAX(date) FROM raw.{table}').fetchone()
    print(f'{table}: {result[0]}')

conn.close()
"
```

### Check Trigger.dev Activity

```bash
# View recent runs
npx trigger.dev@latest runs list --limit 10
```

---

## 🔧 ADDING MORE DATA

### Add New Weather Location

```sql
-- Insert into reference.weather_location_registry
INSERT INTO reference.weather_location_registry VALUES
('US-NE-LNK', 'USW00014939', 'US-NE', 40.850, -96.758, 363, 'Lincoln, NE', TRUE);
```

Next scheduled run will automatically include it.

### Add New Symbol to Databento

Edit `trigger/DataBento/Scripts/collect_daily.py`:
- Add symbol to `SYMBOLS` list
- Redeploy: `npx trigger.dev@latest deploy`

### Add New Data Source

1. Create ingestion script in `trigger/{Source}/Scripts/`
2. Create Trigger.dev job in `trigger/{Source}/{source}_job.ts`
3. Add to `trigger/Orchestration/daily_refresh.ts` batch
4. Deploy: `npx trigger.dev@latest deploy`

---

## ⚠️ CRITICAL RULES

- ✅ **NEVER delete from schemas** - all 77 tables are intentional
- ✅ **NEVER change Big 8 buckets** - preserve all 8
- ✅ **Use weather_location_registry** - don't hardcode locations
- ✅ **Add new data via INSERTs** - don't drop/recreate
- ✅ **Test locally first** - `npx trigger.dev@latest dev`

---

## 📈 NEXT PHASE: AnoFox Features & Training

Once data is populated:
1. Build staging tables (ETL from raw)
2. Run AnoFox SQL macros (feature engineering)
3. Train AutoGluon models (8 bucket specialists + TS core)
4. Publish forecasts to `forecasts.zl_predictions`
5. Dashboard displays live predictions

See `docs/architecture/MASTER_PLAN.md` for full training architecture.
