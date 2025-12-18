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

### GitHub Actions Scheduling (Ready)
- **Hourly**: Databento updates
- **Daily**: Full refresh at 2 AM UTC (Databento, FRED, NOAA weather)
- **Coordinator pattern**: Parallel execution with real-time progress

### Dashboard (Live)
- Live API endpoint: `/api/live/databento`
- Auto-refresh: Every 5 minutes
- Charts: ForecastFanChart + Big8Panel

---

## 🚀 DEPLOYMENT STEPS

### 1. Configure GitHub Actions Secrets

**CRITICAL**: Scheduled ingestion runs in GitHub Actions (cloud), not on your local machine.

Go to: GitHub repo → Settings → Secrets and variables → Actions

Add these for ALL environments (Development, Staging, Production):

```
MOTHERDUCK_TOKEN=your_motherduck_token
MOTHERDUCK_DB=cbi_v15
DATABENTO_API_KEY=your_databento_key
FRED_API_KEY=your_fred_key
EIA_API_KEY=your_eia_key
NOAA_API_TOKEN=your_noaa_token
```

**Reference:** GitHub Actions secrets (repo settings)

### 2. Enable Scheduled Workflows

```bash
cd /Volumes/Satechi\ Hub/CBI-V15
	# Schedules are defined in `.github/workflows/data_ingestion.yml`
```

Scheduling is defined in `.github/workflows/data_ingestion.yml` (cron + manual dispatch).

Workflows execute on GitHub-hosted runners (`ubuntu-latest` by default).

### 3. Verify Workflow Runs

Go to: GitHub repo → Actions → “Data Ingestion”

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

### Check GitHub Actions Activity

See GitHub Actions run history in the repo UI.

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

Edit `src/ingestion/databento/collect_daily.py`:
- Add symbol to `SYMBOLS` list
- Scheduled workflow runs pick it up automatically (or run locally)

### Add New Data Source

1. Create ingestion script in `src/ingestion/{source}/`
2. Add it to `.github/workflows/data_ingestion.yml` (or local cron)

---

## ⚠️ CRITICAL RULES

- ✅ **NEVER delete from schemas** - all 77 tables are intentional
- ✅ **NEVER change Big 8 buckets** - preserve all 8
- ✅ **Use weather_location_registry** - don't hardcode locations
- ✅ **Add new data via INSERTs** - don't drop/recreate
- ✅ **Test locally first** - run the Python script directly

---

## 📈 NEXT PHASE: AnoFox Features & Training

Once data is populated:
1. Build staging tables (ETL from raw)
2. Run AnoFox SQL macros (feature engineering)
3. Train AutoGluon models (8 bucket specialists + TS core)
4. Publish forecasts to `forecasts.zl_predictions`
5. Dashboard displays live predictions

See `docs/architecture/MASTER_PLAN.md` for full training architecture.
