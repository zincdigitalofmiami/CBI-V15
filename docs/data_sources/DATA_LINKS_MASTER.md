# 🔗 CBI-V15 Data Links & API Endpoints - Master List

## **Last Updated:** December 10, 2025

## 🔑 API Keys & Credentials Required

| Service                       | Env Var                                    | Status    | Cost                                                     |
| ----------------------------- | ------------------------------------------ | --------- | -------------------------------------------------------- |
| **MotherDuck**                | `MOTHERDUCK_TOKEN`                         | ✅ Active | Cloud data warehouse                                     |
| **MotherDuck (Read Scaling)** | `MOTHERDUCK_READ_SCALING_TOKEN`            | Optional  | Read replicas                                            |
| **Databento**                 | `DATABENTO_API_KEY`                        | ✅ Active | Market data subscription (~$50-500/mo)                   |
| **FRED**                      | `FRED_API_KEY`                             | ✅ Active | Free (register at fred.stlouisfed.org)                   |
| **EIA**                       | `EIA_API_KEY`                              | ✅ Active | Free (register at eia.gov)                               |
| **NOAA**                      | `NOAA_API_TOKEN`                           | ✅ Active | Free (register at ncdc.noaa.gov)                         |
| **ScrapeCreators**            | `SCRAPECREATORS_API_KEY`                   | ✅ Active | News/sentiment subscription                              |
| **ProFarmer**                 | `PROFARMER_USERNAME`, `PROFARMER_PASSWORD` | ✅ Active | **PAID** (~$500/mo, CRITICAL source)                     |
| **TradingEconomics**          | `TRADINGECONOMICS_API_KEY`                 | ⚠️ Needed | **PAID** (~$200/mo, high-value data)                     |
| **Anchor**                    | `ANCHOR_API_KEY`                           | ✅ Active | Browser automation for scraping                          |
| **Trigger.dev**               | `TRIGGER_SECRET_KEY`                       | ✅ Active | Job orchestration                                        |
| **OpenAI**                    | `OPENAI_API_KEY`                           | Optional  | LLM utilities (optional, not required for core pipeline) |
| **Gemini**                    | `GOOGLE_API_KEY`                           | Optional  | Alternative LLM                                          |

---

## 📊 1. Market Data (Databento)

### API Endpoint

```
https://api.databento.com
```

### Documentation

- **Main Docs:** https://docs.databento.com/
- **API Reference:** https://docs.databento.com/api-reference-historical
- **Python SDK:** https://github.com/databento/databento-python

### Coverage (38 Futures Symbols)

#### Agricultural/Softs (12 symbols)

```python
ZL  # Soybean Oil (PRIMARY TARGET)
ZS  # Soybeans
ZM  # Soybean Meal
ZC  # Corn
ZW  # Wheat
ZO  # Oats
ZR  # Rough Rice
HE  # Lean Hogs
LE  # Live Cattle
GF  # Feeder Cattle
FCPO  # Crude Palm Oil (Bursa Malaysia) - CRITICAL
```

#### Energy (4 symbols)

```python
CL  # WTI Crude Oil
HO  # Heating Oil / ULSD
RB  # RBOB Gasoline
NG  # Natural Gas
```

#### Metals (5 symbols)

```python
HG  # Copper
GC  # Gold
SI  # Silver
PL  # Platinum
PA  # Palladium
```

#### FX Futures (10 symbols)

```python
6E  # Euro FX
6J  # Japanese Yen
6B  # British Pound
6C  # Canadian Dollar
6A  # Australian Dollar
6N  # New Zealand Dollar
6M  # Mexican Peso
6L  # Brazilian Real
6S  # Swiss Franc
DX  # U.S. Dollar Index
```

#### Treasuries (4 symbols)

```python
ZF  # 5-Year Treasury Note
ZN  # 10-Year Treasury Note
ZB  # 30-Year Treasury Bond
```

### Data Fields

- OHLCV (Open, High, Low, Close, Volume)
- Tick data (trade-level)
- Bid/Ask spreads
- Continuous contracts

### Rate Limits

- 1,000 requests per minute

### Ingestion Schedule

- **ZL (Primary):** Every hour
- **All others:** Every 4 hours

### Scripts

```bash
# Python script (legacy)
python trigger/DataBento/Scripts/collect_daily.py

# Trigger.dev job (canonical)
trigger/DataBento/Scripts/databento_ingest_job.ts
```

---

## 📈 2. Economic Data (FRED)

### API Endpoint

```
https://api.stlouisfed.org/fred/
```

### Documentation

- **Main:** https://fred.stlouisfed.org/docs/api/fred/
- **Series Search:** https://fred.stlouisfed.org/tags/series
- **Register:** https://fred.stlouisfed.org/api/signup

### Key Series (24 indicators)

#### Interest Rates & Yields

```
FEDFUNDS    # Federal Funds Rate
DGS1MO      # 1-Month Treasury
DGS3MO      # 3-Month Treasury
DGS6MO      # 6-Month Treasury
DGS1        # 1-Year Treasury
DGS2        # 2-Year Treasury
DGS5        # 5-Year Treasury
DGS7        # 7-Year Treasury
DGS10       # 10-Year Treasury
DGS20       # 20-Year Treasury
DGS30       # 30-Year Treasury
```

#### Yield Spreads

```
T10Y2Y      # 10Y-2Y Spread
T10Y3M      # 10Y-3M Spread
TEDRATE     # TED Spread
```

#### Financial Conditions

```
NFCI        # National Financial Conditions Index
STLFSI4     # St. Louis Fed Financial Stress Index
```

#### Economic Indicators

```
UNRATE      # Unemployment Rate
CPIAUCSL    # CPI (All Urban Consumers)
GDP         # Gross Domestic Product
PAYEMS      # Nonfarm Payrolls
```

#### Market Indicators

```
VIXCLS      # VIX (Volatility Index)
DTWEXBGS    # Dollar Index (Broad)
DTWEXAFEGS  # Dollar Index (Advanced Foreign Economies)
DTWEXEMEGS  # Dollar Index (Emerging Markets)
```

### Rate Limits

- 120 requests per minute

### Target Table

`raw.fred_economic` (single table, not split into multiple tables)

**Note:** SQL macros reference `raw.fred_economic` directly. Plan originally proposed splitting into 3 tables (`raw.fred_rates_spreads`, `raw.fred_financial_conditions`, `raw.fred_real_economy`), but actual implementation uses single table.

### Trigger Jobs

- `trigger/FRED/Scripts/fred_seed_harvest.ts` → `raw.fred_economic` ✅ **CREATED**

**Status:** ✅ **Active** - Single job writes to `raw.fred_economic` table

### Ingestion Schedule

- Daily at 1 AM UTC

### Scripts

```bash
# Python scripts (legacy)
python trigger/FRED/Scripts/collect_fred_fx.py
python trigger/FRED/Scripts/collect_fred_rates_curve.py
python trigger/FRED/Scripts/collect_fred_financial_conditions.py

# Trigger.dev job (canonical)
trigger/FRED/Scripts/fred_seed_harvest.ts
```

---

## 🛢️ 3. Biofuels & Energy

### 3.1 EPA RIN Prices (CRITICAL - FREE)

**Source:** EPA EMTS (Environmental Management Tracking System)

**URLs:**

- **Primary:** https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information
- **Public Data:** https://www.epa.gov/fuels-registration-reporting-and-compliance-help/public-data-renewable-fuel-standard
- **Backup:** https://growthenergy.org/data-set-category/rin-prices/

**Data:** Weekly volume-weighted average RIN prices (D3, D4, D5, D6)

**Coverage:** July 2010 - Present

**Target Table:** `raw.epa_rin_prices` (CRITICAL - referenced in SQL macros)

**Trigger Job:** `trigger/EIA_EPA/Scripts/epa_rin_prices.ts` ⚠️ **Needs creation**

**Frequency:** Weekly (updated monthly by EPA)

**Status:** ✅ **Table definition exists** - `database/models/01_raw/epa_rin_prices.sql` ✅ **CREATED**

### 3.2 EIA (Energy Information Administration)

**API Endpoint:**

```
https://api.eia.gov/v2/
```

**Documentation:**

- **Main:** https://www.eia.gov/opendata/
- **API Guide:** https://www.eia.gov/opendata/documentation.php
- **Register:** https://www.eia.gov/opendata/register.php

**Key Series (6 indicators):**

#### Biofuel Production

```
BIODIESEL_PROD          # Monthly biodiesel production
BIODIESEL_CONSUMPTION   # Monthly consumption
RFS_VOLUMES             # Renewable Volume Obligations
```

#### Petroleum

```
ULSD_WHOLESALE_MIDWEST  # Ultra-low sulfur diesel prices
```

**Rate Limits:**

- No official limit (be respectful)

**Ingestion Schedule:**

- Daily at 1 AM UTC

**Scripts:**

```bash
# Python script (legacy)
python trigger/EIA_EPA/Scripts/collect_eia_biofuels.py

# Trigger.dev job (canonical)
trigger/EIA_EPA/Scripts/eia_procurement_ingest.ts ✅ **CREATED**
```

---

## 📣 4. News & Sentiment (ScrapeCreators)

### API Endpoint

```
https://api.scrapecreators.com/
```

### Documentation

- Contact: support@scrapecreators.com
- Custom buckets configured for CBI-V15

### Available Buckets (8 total)

#### Active Buckets

```python
biofuel_policy      # RFS, biodiesel mandates, EPA rules
china_demand        # Import demand signals, trade data
tariffs_trade       # Trade policy, tariffs, export restrictions
trump_truth_social  # Trump Truth Social posts
```

#### Planned Buckets

```python
market_volatility   # VIX, risk-off sentiment
crop_failures       # Weather, disease, pests
supply_chain        # Logistics, shipping, port data
general_market      # General commodity news
```

### Data Fields

- Sentiment score (-1 to +1)
- Article count
- Keyword frequency
- Source reliability
- Publication timestamp

### Rate Limits

- 60 requests per minute

### Ingestion Schedule

- Every 15 minutes (real-time monitoring)

### Scripts

```bash
# Python scripts (legacy)
python trigger/ScrapeCreators/Scripts/collect_news_buckets.py
python trigger/ScrapeCreators/Scripts/buckets/collect_biofuel_policy.py
python trigger/ScrapeCreators/Scripts/buckets/collect_china_demand.py
python trigger/ScrapeCreators/Scripts/buckets/collect_tariffs_trade_policy.py
python trigger/ScrapeCreators/Scripts/buckets/collect_trump_truth_social.py

# Trigger.dev jobs (canonical)
trigger/ScrapeCreators/Scripts/intelligent_news_pipeline.ts ✅ **CREATED**
trigger/ScrapeCreators/Scripts/news_to_signals_openai_agent.ts ✅ **CREATED**
```

---

## 🌾 5. USDA Data

### API Endpoints

#### NASS QuickStats

```
https://quickstats.nass.usda.gov/api
```

#### FAS (Foreign Ag Service)

```
https://apps.fas.usda.gov/OpenData/
```

### Documentation

- **QuickStats:** https://quickstats.nass.usda.gov/api
- **FAS:** https://www.fas.usda.gov/data
- **API Key:** https://quickstats.nass.usda.gov/api

### Key Data Series (10+)

#### WASDE (World Agricultural Supply & Demand)

```
World Soybean Oil Production
World Soybean Meal Production
World Soybean Production
U.S. Soybean Stocks
```

#### Export Sales (Weekly)

```
Soybeans - Weekly Export Sales
Soybean Oil - Weekly Export Sales
Soybean Meal - Weekly Export Sales
```

#### Crop Progress & Conditions

```
Soybeans - % Planted
Soybeans - % Emerged
Soybeans - Good/Excellent Ratings
Corn - % Planted
Corn - Good/Excellent Ratings
```

### Status

⚠️ **Pending:** Ingestion pipeline not yet built

### Trigger Jobs

- `trigger/USDA/Scripts/usda_fas_exports.ts` → `raw.usda_export_sales` ⚠️ **Needs creation**
- `trigger/USDA/Scripts/usda_wasde.ts` → `raw.usda_wasde` ⚠️ **Needs creation**
- `trigger/USDA/Scripts/usda_nass_quickstats.ts` → `raw.usda_nass` ⚠️ **Needs creation**

### Scripts

```bash
# Python scripts (legacy)
python trigger/USDA/Scripts/ingest_wasde.py        # Monthly WASDE reports
python trigger/USDA/Scripts/ingest_export_sales.py # Weekly export data
python trigger/USDA/Scripts/usda_nass_quickstats.py # Weekly crop status
```

**Status:** ✅ **Jobs created** - No mock data, all real data sources

---

## 🌦️ 6. Weather Data (NOAA)

### API Endpoints

#### NOAA GFS (Global Forecast System)

```
https://nomads.ncep.noaa.gov/
```

#### GSOD (Global Summary of the Day)

```
https://www.ncei.noaa.gov/access/services/data/v1
```

### Documentation

- **NOAA API:** https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- **GFS:** https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
- **GSOD:** https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516

### Coverage (14 regions)

#### Brazil (6 regions)

```
Mato Grosso
Goiás
Mato Grosso do Sul
Paraná
Rio Grande do Sul
Bahia
```

#### Argentina (4 regions)

```
Buenos Aires
Córdoba
Santa Fe
Entre Ríos
```

#### United States (4 regions)

```
Eastern Corn Belt (IL, IN, OH)
Western Corn Belt (IA, MN, NE)
Northern Plains (ND, SD)
Central Plains (KS, NE)
```

### Data Fields

- Temperature (min, max, avg)
- Precipitation
- Drought indices
- Growing degree days

### Status

⚠️ **Partial:** Legacy weather scripts existed in `src/ingestion/legacy_weather/`; canonical NOAA ingestion is now `trigger/Weather/Scripts/ingest_weather.py`.

### Scripts

```bash
# Python scripts (legacy)
python trigger/Weather/Scripts/ingest_weather.py    # NOAA weather data

# Trigger.dev jobs (needed)
trigger/Weather/Scripts/noaa_weather.ts ⚠️ **Needs creation**
trigger/Weather/Scripts/inmet_brazil.ts ⚠️ **Needs creation**
trigger/Weather/Scripts/smn_argentina.ts ⚠️ **Needs creation**
```

---

## 📊 7. CFTC Commitment of Traders

### Data Source

```
https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
```

### Documentation

- **Main:** https://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes/index.htm
- **Data Files:** https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm

### Coverage

- All CME/NYMEX/COMEX futures (38 symbols)
- FX Futures (10 symbols)
- Commodity Futures (24 symbols)
- Treasury Futures (4 symbols)

### Data Fields

```python
Net Positions (Commercial, Non-Commercial, Managed Money)
Long Positions
Short Positions
Open Interest
Spreads
Change from Prior Week
% of Open Interest
```

### Release Schedule

- **Weekly:** Every Tuesday at 3:30 PM ET
- **Data as of:** Prior Tuesday close

### Target Tables

- `raw.cftc_cot_disaggregated` (used by SQL macros)
- `raw.cftc_cot_tff` (Traders in Financial Futures)

### Trigger Job

`trigger/CFTC/Scripts/cftc_cot_reports.ts` ⚠️ **Needs creation**

**Frequency:** Weekly (Friday after report release)

**Status:** ⚠️ **Job needs creation** - Python script exists at `trigger/CFTC/Scripts/ingest_cot.py`

### Scripts

```bash
# Python script (legacy)
python trigger/CFTC/Scripts/ingest_cot.py
```

---

## 🏝️ 8. Vegas Events (Glide API)

### API Endpoint

```
https://api.glide.app/api/v1/
```

### Documentation

- Custom Glide app integration
- Contact: Glide support

### Data Fields

```python
Event ID
Event Name
Event Date
Event Type
Venue
Description
Attendance (estimated)
```

### Rate Limits

- 100 requests per minute

### Ingestion Schedule

- Daily at 3 AM UTC

### Scripts

```bash
# Python script (legacy)
python trigger/Vegas/Scripts/collect_vegas_intel.py

# Trigger.dev job (canonical)
trigger/Vegas/Scripts/vegas_intel_job.ts ✅ **CREATED**
```

---

## 🗄️ 9. MotherDuck (Data Warehouse)

### Connection

```python
import duckdb
conn = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
```

### Web Console

```
https://app.motherduck.com/
```

### Database

```
motherduck-cbi-v15  or  cbi_v15
```

### Schemas

```sql
raw         -- Raw ingestion tables
staging     -- Cleaned/normalized data
features    -- Feature engineering output
training    -- Training datasets
forecasts   -- Model predictions
reference   -- Calendars, catalogs, metadata
ops         -- Pipeline metrics, logs
```

---

## 🌾 10. ProFarmer (CRITICAL - PAID)

### Login URL

```
https://www.profarmer.com
```

### Credentials

```bash
PROFARMER_USERNAME="chris@usoilsolutions.com"
PROFARMER_PASSWORD="*Usoil12025"
```

### Coverage (22+ URLs - Comprehensive)

**Daily Editions:**

- First Thing Today
- Ahead of the Open
- After the Bell

**News Sections:**

- Agriculture News
- Market News
- Policy News
- Weather News

**Newsletters:**

- Weekly Outlook

**Market Analysis:**

- Grains Analysis
- Livestock Analysis
- Energy Analysis

**Commodity Reports:**

- Soybeans
- Soybean Oil
- Soybean Meal
- Corn
- Wheat
- Crude Oil

**Weather:**

- Forecasts
- Crop Conditions

### Target Table

`raw.bucket_news` (source: 'profarmer_all_urls')

### Trigger Job

`trigger/ProFarmer/Scripts/profarmer_all_urls.ts` ✅ **CREATED**

**Frequency:** 3x daily (6 AM, 12 PM, 6 PM UTC)

**Method:** Anchor browser automation (authenticated scraping)

**Buckets:** Crush, China, Biofuel, Weather, Tariff

### Value

**CRITICAL** - ProFarmer is one of the best real-time ag intelligence sources with comprehensive coverage of all market drivers.

### Cost

~$500/month subscription

### Status

✅ Active subscription, credentials in Keychain, comprehensive job created

---

## 📰 11. University of Illinois Intelligence Feeds (FREE - MANDATORY)

### 11.1 Farm Policy News (CRITICAL for China/Tariff)

**URL:**

```
https://farmpolicynews.illinois.edu/
```

**Author:** Keith Good (University of Illinois)

**Why Critical:** Real-time policy news directly impacting soybeans

**Categories:**

- `trade` → China, Tariff buckets
- `ethanol` → Biofuel bucket
- `budget` → Fed bucket
- `regulations` → Tariff, Biofuel buckets
- `immigration` → Policy risk

**Example Headlines:**

- "China Soybean Buying Deadline Now February"
- "$11B Bridge Farm Aid"
- RFS policy updates

**Target Table:** `raw.bucket_news` (source: 'farm_policy_news')

**Trigger Job:** `trigger/UofI_Feeds/Scripts/farmpolicynews.ts` ⚠️ **Needs creation**

**Frequency:** Hourly check for new articles

**Status:** ⚠️ **Job needs creation**

### 11.2 farmdoc Daily (Market Intelligence)

**URLs:**

- **Main:** https://farmdoc.illinois.edu/
- **Daily:** https://farmdocdaily.illinois.edu/

**Categories:**

- `biofuels/rins` → Biofuel bucket (Scott Irwin D4 RIN pricing)
- `agricultural-policy/trade` → China, Tariff buckets (Carl Zulauf)
- `marketing-and-outlook/grain-outlook` → Crush bucket (Nick Paulson)
- `finance/interest-rates` → Fed bucket (Michael Langemeier)
- `marketing-and-outlook/weekly-outlook` → All buckets

**Target Table:** `raw.bucket_news` (source: 'farmdoc_daily')

**Trigger Job:** `trigger/UofI_Feeds/Scripts/farmdoc_daily.ts` ⚠️ **Needs creation**

**Frequency:** Daily at 6 AM UTC

**Status:** ⚠️ **Job needs creation**

---

## 🌍 12. TradingEconomics (HIGH - PAID)

### API Endpoint

```
https://api.tradingeconomics.com/
```

### Documentation

- **Main:** https://docs.tradingeconomics.com/
- **Register:** https://tradingeconomics.com/analytics/api.aspx

### Coverage

- Global commodity prices
- Economic calendars
- Central bank rates
- Trade flow data

### Credentials

```bash
TRADINGECONOMICS_API_KEY="your_key_here"  # ⚠️ Need to obtain
```

### Cost

~$200/month for API access

### Status

⚠️ Keychain placeholder created, need actual API key

---

## 📝 Quick Reference: All API Keys

### Environment Variables

```bash
# Core Database
export MOTHERDUCK_TOKEN="your_token_here"
export MOTHERDUCK_DB="cbi_v15"
export MOTHERDUCK_READ_SCALING_TOKEN="your_read_token"  # Optional

# Market Data
export DATABENTO_API_KEY="your_key_here"

# Economic Data
export FRED_API_KEY="your_key_here"
export EIA_API_KEY="your_key_here"
export NOAA_API_TOKEN="your_token_here"

# News & Sentiment
export SCRAPECREATORS_API_KEY="your_key_here"

# Premium Sources (PAID)
export PROFARMER_USERNAME="your_username"
export PROFARMER_PASSWORD="your_password"
export TRADINGECONOMICS_API_KEY="your_key_here"

# Automation
export ANCHOR_API_KEY="your_key_here"
export TRIGGER_SECRET_KEY="your_trigger_key"

# AI/ML
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-5.1"  # Or custom model

# USDA (if using API)
export USDA_NASS_API_KEY="your_key_here"
```

### Store in `.env` file

```bash
# Copy template
cp config/env-templates/env.template .env

# Edit with your keys
nano .env

# Load in shell
source .env
```

---

## 🔧 Testing Connections

### Test All APIs

```bash
python scripts/ops/test_connections.py
```

### Test Individual Sources

```bash
# MotherDuck
python -c "import duckdb; print(duckdb.connect('md:').execute('SELECT 1').fetchone())"

# Databento
python -c "import databento as db; client = db.Historical(); print('Databento OK')"

# FRED
curl "https://api.stlouisfed.org/fred/series?series_id=FEDFUNDS&api_key=$FRED_API_KEY&file_type=json"

# EIA
curl "https://api.eia.gov/v2/seriesid/RIN_D4?api_key=$EIA_API_KEY"
```

---

## 📊 Data Ingestion Schedule

| Source                 | Frequency | Time (UTC)        | Trigger Job                                                     | Status            |
| ---------------------- | --------- | ----------------- | --------------------------------------------------------------- | ----------------- |
| **Databento (ZL)**     | Hourly    | Every hour        | `trigger/DataBento/Scripts/databento_ingest_job.ts`             | ✅ **CREATED**    |
| **Databento (Others)** | Every 4h  | 0,4,8,12,16,20    | Same                                                            | ✅ **CREATED**    |
| **FRED**               | Daily     | 1 AM              | `trigger/FRED/Scripts/fred_seed_harvest.ts`                     | ✅ **CREATED**    |
| **EPA RIN Prices**     | Weekly    | Mon 8 AM          | `trigger/EIA_EPA/Scripts/epa_rin_prices.ts`                     | ⚠️ Needs creation |
| **EIA**                | Daily     | 1 AM              | `trigger/EIA_EPA/Scripts/eia_procurement_ingest.ts`             | ✅ **CREATED**    |
| **ScrapeCreators**     | 15 min    | Continuous        | `trigger/ScrapeCreators/Scripts/intelligent_news_pipeline.ts`   | ✅ **CREATED**    |
| **ProFarmer**          | 3x daily  | 6 AM, 12 PM, 6 PM | `trigger/ProFarmer/Scripts/profarmer_all_urls.ts`               | ✅ **CREATED**    |
| **Farm Policy News**   | Hourly    | Every hour        | `trigger/UofI_Feeds/Scripts/farmpolicynews.ts`                  | ⚠️ Needs creation |
| **farmdoc Daily**      | Daily     | 6 AM              | `trigger/UofI_Feeds/Scripts/farmdoc_daily.ts`                   | ⚠️ Needs creation |
| **Vegas Events**       | Daily     | 3 AM              | `trigger/Vegas/Scripts/vegas_intel_job.ts`                      | ✅ **CREATED**    |
| **USDA Export Sales**  | Weekly    | Thu 8 AM          | `trigger/USDA/Scripts/usda_fas_exports.ts`                      | ⚠️ Needs creation |
| **USDA WASDE**         | Monthly   | Report date       | `trigger/USDA/Scripts/usda_wasde.ts`                            | ⚠️ Needs creation |
| **CFTC COT**           | Weekly    | Fri 8 PM          | `trigger/CFTC/Scripts/cftc_cot_reports.ts`                      | ⚠️ Needs creation |
| **Weather**            | Daily     | 2 AM              | `trigger/Weather/Scripts/noaa_weather.ts`                       | ⚠️ Needs creation |
| **TradingEconomics**   | Daily     | 1 AM              | `trigger/TradingEconomics/Scripts/tradingeconomics_goldmine.ts` | ✅ **CREATED**    |

---

## ✅ Summary

### Active Data Sources (7)

1. ✅ **Databento** - 38 futures symbols (OHLCV + tick)
2. ✅ **FRED** - 24 macro indicators
3. ✅ **EIA** - Biofuel production, petroleum data
4. ✅ **EPA RIN Prices** - Weekly D3/D4/D5/D6 prices (FREE, CRITICAL)
5. ✅ **ScrapeCreators** - 4 active news buckets
6. ✅ **ProFarmer** - Comprehensive ag intelligence (22+ URLs, 3x daily)
7. ✅ **MotherDuck** - Data warehouse

### Pending Data Sources (6)

8. ⚠️ **USDA** - WASDE, export sales, crop data (Trigger jobs needed)
9. ⚠️ **CFTC** - Commitment of Traders (Trigger job needed)
10. ⚠️ **NOAA** - Weather (14 regions) (Trigger jobs needed)
11. ⚠️ **Farm Policy News** - University of Illinois (Trigger job needed)
12. ⚠️ **farmdoc Daily** - University of Illinois (Trigger job needed)
13. ⚠️ **TradingEconomics** - Global commodity data (API key needed)

### Total Coverage

- **38 Futures Symbols** (Databento)
- **24 Macro Indicators** (FRED)
- **14 Weather Regions** (NOAA, INMET, SMN)
- **8 News Buckets** (ScrapeCreators: 4 active, 4 planned)
- **3 University Sources** (ProFarmer, Farm Policy News, farmdoc Daily)
- **EPA RIN Prices** (D3, D4, D5, D6 weekly)
- **6 Biofuel Series** (EIA)

### Trigger.dev Orchestration

All ingestion jobs organized by **source** in `trigger/<Source>/Scripts/`:

- `trigger/DataBento/Scripts/` - Databento futures
- `trigger/FRED/Scripts/` - FRED economic data
- `trigger/EIA_EPA/Scripts/` - EIA biofuels, EPA RIN prices
- `trigger/USDA/Scripts/` - USDA WASDE, export sales, crop data
- `trigger/CFTC/Scripts/` - CFTC COT positioning data
- `trigger/Weather/Scripts/` - NOAA, INMET, SMN weather
- `trigger/ProFarmer/Scripts/` - ProFarmer premium ag intelligence
- `trigger/UofI_Feeds/Scripts/` - Farm Policy News, farmdoc Daily
- `trigger/ScrapeCreators/Scripts/` - News buckets, sentiment
- `trigger/Vegas/Scripts/` - Vegas events intel
- `trigger/TradingEconomics/Scripts/` - Global commodity data
- `trigger/Policy/Scripts/` - Government & think tank policy docs
- `trigger/Analysts/Scripts/` - Analyst feeds, social media

---

## 🌦️ 13. Weather Data - Expanded Coverage

### 13.1 INMET Brazil (Institutional-Grade Weather)

**API Endpoints:**

```
https://apitempo.inmet.gov.br/estacao/{start}/{end}/{station_id}
https://apitempo.inmet.gov.br/token
https://apitempo.inmet.gov.br/estacoes
https://apitempo.inmet.gov.br/estacao/dados-estacao
https://apitempo.inmet.gov.br/estacoes/T/A/,,/A/
https://portal.inmet.gov.br/api/estacoes/automaticas
```

**Coverage:** Station-level data for all 6 Brazil soybean regions

**Target Table:** `raw.noaa_weather_daily` (unified weather table)

**Status:** ⚠️ **Needs implementation**

### 13.2 Argentina SMN (Servicio Meteorológico Nacional)

**API Endpoint:**

```
https://ssl.smn.gob.ar/dpd/descarga_opendata.php?file=observaciones/datohorario{station_id}.txt
```

**Coverage:** Hourly text data for 4 Argentina soybean regions

**Target Table:** `raw.noaa_weather_daily`

**Status:** ⚠️ **Needs implementation**

### 13.3 Copernicus (ECMWF)

**API Endpoint:**

```
https://cds.climate.copernicus.eu/api
```

**Coverage:** European Centre for Medium-Range Weather Forecasts

**Status:** ⚠️ **Needs implementation**

### 13.4 NOAA NOMADS GFS

**Filter Endpoint:**

```
https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
```

**Coverage:** Global Forecast System 0.25° resolution

**Status:** ⚠️ **Needs implementation**

### 13.5 Meteomatics

**API Endpoint:**

```
https://api.meteomatics.com
```

**Status:** ⚠️ **Requires API key**

---

## 🏦 14. Central Banks & Treasury Data

### 14.1 US Treasury Fiscal Data

**API Endpoint:**

```
https://api.fiscaldata.treasury.gov/services/api/v1/
```

**Coverage:**

- Treasury auctions: `https://www.treasurydirect.gov/auctions/`
- TreasuryDirect: `https://www.treasurydirect.gov/`

**Target Table:** `raw.fred_economic` (treasury data)

**Status:** ⚠️ **Needs implementation**

### 14.2 BLS (Bureau of Labor Statistics)

**API Endpoint:**

```
https://api.bls.gov/publicAPI/v2/
```

**Key Series:**

- UNRATE - Unemployment Rate
- CPIAUCSL - Consumer Price Index
- PAYEMS - Nonfarm Payrolls

**Target Table:** `raw.fred_economic`

**Status:** ⚠️ **Needs implementation**

### 14.3 ECB (European Central Bank)

**API Endpoint:**

```
https://sdw-wsrest.ecb.europa.eu/service/
```

**Coverage:** ECB SDW REST API

**Target Table:** `raw.fred_economic`

**Status:** ⚠️ **Needs implementation**

### 14.4 Banco Central do Brasil (SGS)

**API Endpoint:**

```
https://www3.bcb.gov.br/sgspub/
```

**Coverage:** Brazil monetary policy, FX rates (BRL/USD)

**Target Table:** `raw.fred_economic`

**Status:** ⚠️ **Needs implementation**

### 14.5 PBOC (People's Bank of China)

**URL:**

```
http://www.pbc.gov.cn/en/
```

**Coverage:** China monetary policy announcements

**Status:** ⚠️ **Web scraping required**

### 14.6 BCRA (Argentina Central Bank)

**URL:**

```
http://www.bcra.gob.ar/
```

**Coverage:** Argentina monetary policy, ARS/USD rates

**Status:** ⚠️ **Web scraping required**

---

## 📅 15. Economic Calendars (Real-Time Event Data)

### 15.1 TradingEconomics Calendar

**URL:**

```
https://tradingeconomics.com/calendar
```

**Status:** ✅ **Available via TradingEconomics API** (requires key)

### 15.2 ForexFactory Calendar

**URL:**

```
https://www.forexfactory.com/calendar
```

**Status:** ⚠️ **Web scraping required**

### 15.3 Investing.com Calendar

**URL:**

```
https://www.investing.com/economic-calendar/
```

**Status:** ⚠️ **Web scraping required**

### 15.4 MarketWatch Calendar

**URL:**

```
https://www.marketwatch.com/economy-politics/calendar
```

**Status:** ⚠️ **Web scraping required**

---

## 📈 16. Market Data - Alternative Sources

### 16.1 Polygon.io

**API Endpoint:**

```
https://api.polygon.io
```

**Documentation:**

```
https://polygon.io/docs/stocks/get_v2_aggs_tickerstocksTicker_rangemultipliertimespan_fromto
```

**Coverage:** Stocks, forex, crypto tick data

**Credentials:**

```bash
POLYGON_API_KEY="your_key_here"
```

**Status:** ⚠️ **Available, needs implementation**

### 16.2 Alpha Vantage

**Coverage:** Free market data API (equities, FX, crypto)

**Credentials (from v14):**

```bash
ALPHA_VANTAGE_API_KEY="BA7CQWXKRFBNFY49"
```

**Status:** ⚠️ **Free tier, limited to 5 calls/min**

### 16.3 NY Fed Latest Rates

**API Endpoint:**

```
https://markets.newyorkfed.org/api/rates/all/latest.json
```

**Coverage:** Fed rates in real-time JSON

**Status:** ⚠️ **Free, no auth required**

---

## 🏛️ 17. Policy & Regulatory Sources

### 17.1 Federal Register

**API Endpoint:**

```
https://www.federalregister.gov/api/v1/documents.json
```

**Coverage:** US federal regulations, EPA rules, trade policy

**Target Table:** `raw.bucket_news` (source: 'federal_register')

**Status:** ⚠️ **Needs implementation**

### 17.2 Immigration Enforcement (ICE/DHS/CBP)

**URLs:**

```
https://www.ice.gov/news/releases
https://www.dhs.gov/news-releases
https://www.cbp.gov/newsroom
```

**Relevance:** Labor policy impacts on ag sector

**Target Table:** `raw.bucket_news`

**Status:** ⚠️ **Needs implementation**

### 17.3 Fed Speeches & FOMC

**URLs:**

```
https://www.federalreserve.gov/newsevents/speech/
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
```

**Target Table:** `raw.bucket_news` (source: 'fed_communications')

**Status:** ⚠️ **Needs implementation**

---

## 🌾 18. Farm Organizations & Advocacy

**URLs:**

```
https://www.farmlabororganizing.org/
https://ufw.org/
https://www.wga.com/
https://www.fb.org/newsroom/
https://immigrationimpact.com/
https://www.migrationpolicy.org/
https://www.splcenter.org/issues/immigrant-justice
https://www.cfbf.com/news/
https://www.texasagriculture.gov/
https://www.fdacs.gov/
https://www.gfb.org/
```

**Target Table:** `raw.bucket_news` (source: 'farm_advocacy')

**Status:** ⚠️ **Needs implementation**

---

## 🎯 19. Think Tanks & Trade Policy

**URLs:**

```
https://www.heritage.org/agriculture
https://americafirstpolicy.com/
https://taxfoundation.org/research/all/federal/trade/
https://www.aei.org/tag/trade-policy/
https://www.piie.com/research/piie-charts/us-china-trade-war-tariffs-date-chart
https://www.csis.org/programs/scholl-chair-international-business/trade-war-monitor
https://www.uschina.org/
```

**Target Table:** `raw.bucket_news` (source: 'think_tank_policy')

**Status:** ⚠️ **Needs implementation**

---

## 🗳️ 20. Campaign & Political Sources

**URLs:**

```
https://www.donaldjtrump.com/news
https://www.winred.com/save-america-joint-fundraising-committee/
```

**Target Table:** `raw.bucket_news` (source: 'campaign_political')

**Status:** ⚠️ **Needs implementation**

---

## 📰 21. Ag/Commodities Media

**URLs:**

```
https://www.agweb.com/news/crops/soybeans
https://www.farmprogress.com/soybeans
https://www.agriculture.com/markets-commodities
https://www.agrimoney.com/news/grains-oilseeds/
https://www.world-grain.com/
https://www.conab.gov.br/ultimas-noticias
https://abiove.org.br/en/statistics/
https://www.agrimoney.com/news/china/
https://www.reuters.com/business/commodities/
https://www.eia.gov/biofuels/biodiesel/production/
```

**Target Table:** `raw.bucket_news` (source: 'ag_media')

**Status:** ⚠️ **Needs implementation**

---

## 📱 22. Social Media Intelligence

### 22.1 Truth Social (via ScrapeCreators)

**Endpoints:**

```
https://api.scrapecreators.com/v1/truthsocial
https://api.scrapecreators.com/v1/truthsocial/post
https://truthsocial.com/@realDonaldTrump
https://truthsocial.com/@realDonaldTrump/{post_id}
```

**Auth (from v14):**

```bash
SCRAPECREATORS_API_KEY="B1TOgQvMVSV6TDglqB8lJ2cirqi2"
```

**Status:** ✅ **Implemented** in `trigger/ScrapeCreators/Scripts/buckets/collect_trump_truth_social.py`

### 22.2 Facebook Pages (via ScrapeCreators)

**Endpoint:**

```
https://api.scrapecreators.com/v1/facebook/post
```

**Target Pages:**

```
https://www.facebook.com/AmericanSoybeanAssociation/
https://www.facebook.com/departmentoflabor
https://www.facebook.com/USDA
```

**Status:** ⚠️ **Needs implementation**

### 22.3 Reddit Agriculture

**JSON Feed:**

```
https://www.reddit.com/r/agriculture.json
```

**Target Table:** `raw.bucket_news` (source: 'reddit_agriculture')

**Status:** ⚠️ **Needs implementation**

---

## 🚢 23. Shipping & Logistics

### 23.1 MarineTraffic

**URL:**

```
https://www.marinetraffic.com/en/data/
```

**Coverage:** Vessel tracking, port congestion, grain shipping routes

**Target Table:** `raw.shipping_logistics` ⚠️ **Table needs creation**

**Status:** ⚠️ **Requires API subscription**

---

## 🔑 API Keys from V14 Archive (SECURITY AUDIT)

**⚠️ CRITICAL: Hardcoded secrets detected in v14 - MUST rotate**

### Found in v14 Code:

```python
# ScrapeCreators (EXPOSED)
SCRAPECREATORS_API_KEY = "B1TOgQvMVSV6TDglqB8lJ2cirqi2"

# FRED (EXPOSED)
FRED_API_KEY = "dc195c8658c46ee1df83bcd4fd8a690b"

# Alpha Vantage (EXPOSED)
ALPHA_VANTAGE_API_KEY = "BA7CQWXKRFBNFY49"

# NOAA (EXPOSED)
NOAA_TOKEN = "rxoLrCxYOlQyWvVjbBGRlMMhIRElWKZi"
```

**ACTION REQUIRED:**

1. Verify these keys still work
2. Store in `.env` and macOS Keychain (NOT in code)
3. Update all scripts to use `os.getenv()` or keychain_manager
4. Consider rotating if they've been in public repos

---

## 📊 COMPLETE DATA SOURCE INVENTORY (50+ Sources)

### **TIER 1: Core Market Data (4 sources)**

1. Databento - 38 futures symbols
2. CFTC - COT positioning for all 38
3. Polygon.io - Alternative market data
4. NY Fed - Real-time rates

### **TIER 2: Macro/Economic (13 sources)**

5. FRED - 24 US indicators
6. US Treasury - Fiscal data, auctions
7. BLS - Employment, CPI
8. ECB - European rates
9. Banco Central do Brasil - Brazil rates
10. PBOC - China rates
11. BCRA - Argentina rates
12. Fed Speeches - Policy signals
13. FOMC Calendars - Meeting dates
14. TradingEconomics - Global macro
15. ForexFactory Calendar
16. Investing.com Calendar
17. MarketWatch Calendar

### **TIER 3: Weather (9 sources)**

18. NOAA - US weather (14 regions)
19. INMET Brazil - Station-level
20. Argentina SMN - Hourly data
21. Copernicus/ECMWF - European forecasts
22. NOAA NOMADS GFS - Global models
23. Meteomatics - High-resolution forecasts
24. ProFarmer Weather - Ag-specific analysis

### **TIER 4: Biofuels/Energy (3 sources)**

25. EPA - RIN prices (D3/D4/D5/D6)
26. EIA - Biofuel production, petroleum

### **TIER 5: USDA/Supply (4 sources)**

27. USDA WASDE - Monthly supply/demand
28. USDA FAS - Export sales
29. USDA NASS - Crop progress, conditions
30. USDA Price Basis - Regional basis spreads

### **TIER 6: News/Intelligence (15+ sources)**

31. ScrapeCreators - 8 buckets (4 active)
32. ProFarmer - Premium ag intelligence (22 URLs)
33. Farm Policy News - UofI (Keith Good)
34. farmdoc Daily - UofI (Scott Irwin, Carl Zulauf)
35. Federal Register - Regulatory policy
36. ICE/DHS/CBP - Immigration enforcement
37. Trump Campaign - Political news
38. AgWeb - Crop news
39. Farm Progress - Ag markets
40. Agriculture.com - Market analysis
41. Agrimoney - Global grains
42. World Grain - Industry news
43. CONAB Brazil - Brazil production
44. ABIOVE - Brazil exports
45. Reuters Commodities - Breaking news

### **TIER 7: Policy/Advocacy (10+ sources)**

46. Heritage Foundation - Ag policy
47. America First Policy - Trade policy
48. Tax Foundation - Trade analysis
49. AEI - Trade policy
50. PIIE - Trade war charts
51. CSIS - Trade war monitor
52. US-China Business Council
53. Farm Labor Organizing
54. United Farm Workers
55. Multiple state farm bureaus

### **TIER 8: Social/Alternative (3 sources)**

56. Truth Social - Trump posts
57. Facebook - Ag organization pages
58. Reddit Agriculture - Social sentiment

### **TIER 9: Shipping/Logistics (1 source)**

59. MarineTraffic - Vessel tracking

---

## 🗂️ TARGET TABLE MAPPING

### Single-Schema Tables

```sql
raw.databento_futures_ohlcv_1d     -- All 38 futures symbols
raw.fred_economic                  -- All FRED, BLS, Treasury, Central Bank data
raw.eia_biofuels                   -- EIA biofuel series
raw.epa_rin_prices                 -- EPA RIN prices (D3/D4/D5/D6)
raw.cftc_cot_disaggregated         -- COT data for all 38 symbols
raw.cftc_cot_tff                   -- Traders in Financial Futures
raw.noaa_weather_daily             -- All weather (NOAA, INMET, SMN, ECMWF)
raw.bucket_news                    -- All news/policy/advocacy sources
raw.scrapecreators_news_buckets    -- ScrapeCreators segmented news
```

### USDA Multi-Table

```sql
raw.usda_export_sales              -- FAS export data
raw.usda_wasde                     -- WASDE reports
raw.usda_crop_progress             -- Crop conditions
raw.usda_price_basis               -- Regional basis
```

### Future Tables (if needed)

```sql
raw.shipping_logistics             -- MarineTraffic vessel data
raw.economic_calendar              -- Event calendar aggregation
```

---

## 🔒 Security Notes

**⚠️ CRITICAL:** v14 had hardcoded API keys in code. All v15 ingestion MUST:

1. Read keys from environment variables or macOS Keychain
2. Never commit keys to git
3. Use `src/utils/keychain_manager.py` for key retrieval
4. Store in `.env` (gitignored)

**Rotate immediately if v14 keys were exposed:**

- FRED API key
- NOAA token
- Alpha Vantage key
- Any keys found in public repos

---

**This is the canonical data links master list. All other data source references should point here.**

**For Trigger.dev orchestration, see:** `trigger/README.md` and `trigger/WEB_SCRAPING_TARGETS_MASTER.md`

**For API key storage, see:** `scripts/setup/store_api_keys.sh`

**For connection testing, see:** `scripts/ops/test_connections.py`
