# 🔗 CBI-V15 Data Links & API Endpoints - Master List

**Last Updated:** December 7, 2024

---

## 🔑 API Keys & Credentials Required

| Service | Env Var | Status | Cost |
|---------|---------|--------|------|
| **MotherDuck** | `MOTHERDUCK_TOKEN` | ✅ Active | Cloud data warehouse |
| **MotherDuck (Read Scaling)** | `MOTHERDUCK_READ_SCALING_TOKEN` | Optional | Read replicas |
| **Databento** | `DATABENTO_API_KEY` | ✅ Active | Market data subscription (~$50-500/mo) |
| **FRED** | `FRED_API_KEY` | ✅ Active | Free (register at fred.stlouisfed.org) |
| **EIA** | `EIA_API_KEY` | ✅ Active | Free (register at eia.gov) |
| **NOAA** | `NOAA_API_TOKEN` | ✅ Active | Free (register at ncdc.noaa.gov) |
| **ScrapeCreators** | `SCRAPECREATORS_API_KEY` | ✅ Active | News/sentiment subscription |
| **ProFarmer** | `PROFARMER_USERNAME`, `PROFARMER_PASSWORD` | ✅ Active | **PAID** (~$500/mo, CRITICAL source) |
| **TradingEconomics** | `TRADINGECONOMICS_API_KEY` | ⚠️ Needed | **PAID** (~$200/mo, high-value data) |
| **Anchor** | `ANCHOR_API_KEY` | ✅ Active | Browser automation for scraping |
| **Trigger.dev** | `TRIGGER_SECRET_KEY` | ✅ Active | Job orchestration |
| **OpenAI** | `OPENAI_API_KEY` | ✅ Active | TSci agent layer (~$20-100/mo) |
| **Gemini** | `GOOGLE_API_KEY` | Optional | Alternative LLM |

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

### Script
```bash
python src/ingestion/databento/collect_daily.py
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

### Ingestion Schedule
- Daily at 1 AM UTC

### Scripts
```bash
python src/ingestion/fred/collect_fred_fx.py
python src/ingestion/fred/collect_fred_rates_curve.py
python src/ingestion/fred/collect_fred_financial_conditions.py
```

---

## 🛢️ 3. Biofuels & Energy (EIA)

### API Endpoint
```
https://api.eia.gov/v2/
```

### Documentation
- **Main:** https://www.eia.gov/opendata/
- **API Guide:** https://www.eia.gov/opendata/documentation.php
- **Register:** https://www.eia.gov/opendata/register.php

### Key Series (6 indicators)

#### RIN Prices
```
RIN_D4      # Biomass Diesel RINs
RIN_D6      # Ethanol RINs
```

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

### Rate Limits
- No official limit (be respectful)

### Ingestion Schedule
- Daily at 1 AM UTC

### Script
```bash
python src/ingestion/eia/collect_eia_biofuels.py
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
python src/ingestion/scrape_creator/collect.py
python src/ingestion/scrape_creator/buckets/collect_biofuel_policy.py
python src/ingestion/scrape_creator/buckets/collect_china_demand.py
python src/ingestion/scrape_creator/buckets/collect_tariffs_trade_policy.py
python src/ingestion/scrape_creator/buckets/collect_trump_truth_social.py
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

### Planned Scripts
```bash
python src/ingestion/usda/collect_wasde.py        # Monthly WASDE reports
python src/ingestion/usda/collect_export_sales.py # Weekly export data
python src/ingestion/usda/collect_crop_progress.py # Weekly crop status
```

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
⚠️ **Partial:** Legacy weather scripts in `src/ingestion/legacy_weather/`

### Planned Scripts
```bash
python src/ingestion/weather/collect_noaa_ghcnd.py  # Station daily data
python src/ingestion/weather/collect_noaa_gfs.py    # Forecast grids
python src/ingestion/weather/collect_inmet_smn.py   # Brazil/Argentina
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

### Status
⚠️ **Pending:** Ingestion pipeline not yet built

### Planned Script
```bash
python src/ingestion/cftc/collect_cot.py
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

### Script
```bash
python scripts/ops/vegas/collect_vegas_events.py
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
tsci        -- TSci agent jobs/runs
```

---

## 🌾 10. ProFarmer (CRITICAL - PAID)

### Login URL
```
https://www.dtnpf.com/agriculture/web/ag/home
```

### Credentials
```bash
PROFARMER_USERNAME="chris@usoilsolutions.com"
PROFARMER_PASSWORD="*Usoil12025"
```

### Coverage
- Weather forecasts (Brazil, Argentina, US Corn Belt)
- Basis prices (local elevator bids)
- Barge rates (river logistics)
- Crop conditions (yield estimates)
- Market commentary

### Value
**CRITICAL** - ProFarmer is one of the best real-time ag intelligence sources.

### Cost
~$500/month subscription

### Status
✅ Active subscription, credentials in Keychain

---

## 🌍 11. TradingEconomics (HIGH - PAID)

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

| Source | Frequency | Time (UTC) | Script |
|--------|-----------|------------|--------|
| **Databento (ZL)** | Hourly | Every hour | `src/ingestion/databento/collect_daily.py` |
| **Databento (Others)** | Every 4h | 0,4,8,12,16,20 | Same |
| **FRED** | Daily | 1 AM | `src/ingestion/fred/collect_*` |
| **EIA** | Daily | 1 AM | `src/ingestion/eia/collect_eia_biofuels.py` |
| **ScrapeCreators** | 15 min | Continuous | `src/ingestion/scrape_creator/collect.py` |
| **Vegas Events** | Daily | 3 AM | `scripts/ops/vegas/collect_vegas_events.py` |
| **USDA** | Weekly | Mon 8 AM | ⚠️ Pending |
| **CFTC** | Weekly | Tue 3:30 PM ET | ⚠️ Pending |
| **Weather** | Daily | 2 AM | ⚠️ Pending |

---

## ✅ Summary

### Active Data Sources (6)
1. ✅ **Databento** - 38 futures symbols (OHLCV + tick)
2. ✅ **FRED** - 24 macro indicators
3. ✅ **EIA** - 6 biofuel series
4. ✅ **ScrapeCreators** - 4 active news buckets
5. ✅ **Glide** - Vegas events
6. ✅ **MotherDuck** - Data warehouse

### Pending Data Sources (3)
7. ⚠️ **USDA** - WASDE, export sales, crop data
8. ⚠️ **NOAA** - Weather (14 regions)
9. ⚠️ **CFTC** - Commitment of Traders

### Total Coverage
- **38 Futures Symbols**
- **24 Macro Indicators**
- **14 Weather Regions**
- **8 News Buckets** (4 active, 4 planned)
- **6 Biofuel Series**

---

**For detailed feature engineering, see:** `docs/data_sources/DATA_SOURCES_MASTER.md`

**For API key storage, see:** `scripts/setup/store_api_keys.sh`

**For connection testing, see:** `scripts/ops/test_connections.py`

