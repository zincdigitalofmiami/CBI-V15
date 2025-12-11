# 🌐 Web Scraping Targets - Master List

**CBI-V15 Intelligence Platform**  
**Last Updated:** December 10, 2025

---

## 🎯 Overview

This document contains ALL web sources we scrape/monitor for ZL (Soybean Oil) forecasting. These are organized by priority, bucket type, and domain.

**Location:** `trigger/` — All scraping jobs live here, organized by source (`<Source>/Guides`, `<Source>/Scripts`).

---

## 🔗 COMPLETE URL REGISTRY

### 📊 1. MARKET DATA (DataBento)

**Folder:** `trigger/DataBento/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| DataBento API | `https://api.databento.com` | OHLCV, tick data | CRITICAL | ✅ Active |
| DataBento Docs | `https://docs.databento.com/` | Reference | - | - |

**Coverage (38 Futures Symbols):**

#### Agricultural/Softs (12 symbols)
```
ZL   # Soybean Oil (PRIMARY TARGET)
ZS   # Soybeans
ZM   # Soybean Meal
ZC   # Corn
ZW   # Wheat
ZO   # Oats
ZR   # Rough Rice
HE   # Lean Hogs
LE   # Live Cattle
GF   # Feeder Cattle
FCPO # Crude Palm Oil (Bursa Malaysia) - CRITICAL
```

#### Energy (4 symbols)
```
CL   # WTI Crude Oil
HO   # Heating Oil / ULSD
RB   # RBOB Gasoline
NG   # Natural Gas
```

#### Metals (5 symbols)
```
HG   # Copper
GC   # Gold
SI   # Silver
PL   # Platinum
PA   # Palladium
```

#### FX Futures (10 symbols)
```
6E   # Euro FX
6J   # Japanese Yen
6B   # British Pound
6C   # Canadian Dollar
6A   # Australian Dollar
6N   # New Zealand Dollar
6M   # Mexican Peso
6L   # Brazilian Real
6S   # Swiss Franc
DX   # U.S. Dollar Index
```

#### Treasuries (4 symbols)
```
ZF   # 5-Year Treasury Note
ZN   # 10-Year Treasury Note
ZB   # 30-Year Treasury Bond
```

---

### 📈 2. ECONOMIC DATA (FRED)

**Folder:** `trigger/FRED/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| FRED API | `https://api.stlouisfed.org/fred/` | Macro indicators | CRITICAL | ✅ Active |
| FRED Series Search | `https://fred.stlouisfed.org/tags/series` | Reference | - | - |

**Key Series (24 indicators):**

#### Interest Rates & Yields (11)
```
FEDFUNDS   # Federal Funds Rate
DGS1MO     # 1-Month Treasury
DGS3MO     # 3-Month Treasury
DGS6MO     # 6-Month Treasury
DGS1       # 1-Year Treasury
DGS2       # 2-Year Treasury
DGS5       # 5-Year Treasury
DGS7       # 7-Year Treasury
DGS10      # 10-Year Treasury
DGS20      # 20-Year Treasury
DGS30      # 30-Year Treasury
```

#### Yield Spreads (3)
```
T10Y2Y     # 10Y-2Y Spread
T10Y3M     # 10Y-3M Spread
TEDRATE    # TED Spread
```

#### Financial Conditions (2)
```
NFCI       # National Financial Conditions Index
STLFSI4    # St. Louis Fed Financial Stress Index
```

#### Economic Indicators (4)
```
UNRATE     # Unemployment Rate
CPIAUCSL   # CPI (All Urban Consumers)
GDP        # Gross Domestic Product
PAYEMS     # Nonfarm Payrolls
```

#### Market Indicators (4)
```
VIXCLS     # VIX (Volatility Index)
DTWEXBGS   # Dollar Index (Broad)
DTWEXAFEGS # Dollar Index (Advanced Foreign Economies)
DTWEXEMEGS # Dollar Index (Emerging Markets)
```

---

### 🛢️ 3. BIOFUELS & ENERGY (EIA + EPA)

**Folder:** `trigger/EIA_EPA/`

#### EPA RIN Prices (CRITICAL - FREE)

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| EPA RIN Primary | `https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information` | RIN prices | CRITICAL | ⚠️ Needs job |
| EPA Public Data | `https://www.epa.gov/fuels-registration-reporting-and-compliance-help/public-data-renewable-fuel-standard` | RFS data | HIGH | ⚠️ Needs job |
| Growth Energy (backup) | `https://growthenergy.org/data-set-category/rin-prices/` | RIN prices | MEDIUM | Backup |

**RIN Types:** D3, D4, D5, D6 (weekly volume-weighted average)

#### EIA (Energy Information Administration)

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| EIA API v2 | `https://api.eia.gov/v2/` | Energy data | HIGH | ✅ Active |
| EIA Open Data | `https://www.eia.gov/opendata/` | Bulk data | MEDIUM | - |
| EIA Biodiesel | `https://www.eia.gov/biofuels/biodiesel/production/` | Biodiesel stats | HIGH | - |

**Key EIA Series:**
```
BIODIESEL_PROD          # Monthly biodiesel production
BIODIESEL_CONSUMPTION   # Monthly consumption
RFS_VOLUMES             # Renewable Volume Obligations
ULSD_WHOLESALE_MIDWEST  # Ultra-low sulfur diesel prices
```

---

### 🌾 4. USDA DATA

**Folder:** `trigger/USDA/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| NASS QuickStats API | `https://quickstats.nass.usda.gov/api` | Crop data | HIGH | ⚠️ Needs job |
| FAS Open Data | `https://apps.fas.usda.gov/OpenData/` | Export data | CRITICAL | ⚠️ Needs job |
| FAS Export Sales | `https://apps.fas.usda.gov/esrquery/` | Weekly exports | CRITICAL | ⚠️ Needs job |
| FAS GAIN API | `https://apps.fas.usda.gov/newgainapi/` | Trade intel | HIGH | ⚠️ Needs job |
| WASDE Reports | `https://www.usda.gov/oce/commodity/wasde` | Supply/demand | CRITICAL | ⚠️ Needs job |

**WASDE Key Series:**
```
World Soybean Oil Production
World Soybean Meal Production
World Soybean Production
U.S. Soybean Stocks
```

**Export Sales (Weekly):**
```
Soybeans - Weekly Export Sales
Soybean Oil - Weekly Export Sales
Soybean Meal - Weekly Export Sales
```

**Crop Progress & Conditions:**
```
Soybeans - % Planted
Soybeans - % Emerged
Soybeans - Good/Excellent Ratings
Corn - % Planted
Corn - Good/Excellent Ratings
```

---

### 📊 5. CFTC COMMITMENT OF TRADERS

**Folder:** `trigger/CFTC/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| CFTC COT Reports | `https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm` | Positioning | CRITICAL | ✅ Script exists |
| CFTC Historical | `https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm` | Historical | HIGH | - |
| CFTC Explanatory | `https://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes/index.htm` | Reference | - | - |

**Data Fields:**
```
Net Positions (Commercial, Non-Commercial, Managed Money)
Long Positions
Short Positions
Open Interest
Spreads
Change from Prior Week
% of Open Interest
```

**Release:** Weekly on Tuesday 3:30 PM ET

---

### 🌦️ 6. WEATHER DATA

**Folder:** `trigger/Weather/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| NOAA CDO API | `https://www.ncei.noaa.gov/cdo-web/api/v2/data` | Station data | HIGH | ⚠️ Needs job |
| NOAA CDO Services | `https://ncei.noaa.gov/cdo-web/webservices/v2` | API docs | - | - |
| NOAA NOMADS GFS | `https://nomads.ncep.noaa.gov/` | Forecasts | HIGH | ⚠️ Needs job |
| NOAA NOMADS Filter | `https://nomads.noaa.gov/cgi-bin/filter_gfs_0p25.pl` | GFS data | HIGH | - |
| NOAA Token Portal | `https://www.ncdc.noaa.gov/cdo-web/token` | Auth | - | - |
| INMET (Brazil) | `https://apitempo.inmet.gov.br/` | Brazil weather | HIGH | ⚠️ Needs job |
| INMET Token | `https://apitempo.inmet.gov.br/token` | Auth | - | - |
| INMET Stations | `https://apitempo.inmet.gov.br/estacoes` | Station list | - | - |
| INMET Daily | `https://apitempo.inmet.gov.br/estacoes/diarias` | Daily data | HIGH | - |
| INMET Historical | `https://portal.inmet.gov.br/dadoshistoricos` | Historical | MEDIUM | - |
| SMN Argentina | `https://ssl.smn.gob.ar/ddopen2/gare` | Argentina wx | HIGH | ⚠️ Needs job |
| SMN DPD | `https://ssl.smn.gob.ar/dpd/` | Argentina data | HIGH | - |
| Copernicus CDS | `https://cds.climate.copernicus.eu/api` | Climate data | MEDIUM | ⚠️ Needs job |
| Meteomatics | `https://api.meteomatics.com` | Weather API | MEDIUM | - |

**Coverage (14 regions):**

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

---

### 📣 7. NEWS & SENTIMENT (ScrapeCreators)

**Folder:** `trigger/ScrapeCreators/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| ScrapeCreators API | `https://api.scrapecreators.com/` | News buckets | HIGH | ✅ Active |

**Active Buckets (4):**
```
biofuel_policy      # RFS, biodiesel mandates, EPA rules
china_demand        # Import demand signals, trade data
tariffs_trade       # Trade policy, tariffs, export restrictions
trump_truth_social  # Trump Truth Social posts
```

**Planned Buckets (4):**
```
market_volatility   # VIX, risk-off sentiment
crop_failures       # Weather, disease, pests
supply_chain        # Logistics, shipping, port data
general_market      # General commodity news
```

---

### 🌾 8. AGRICULTURAL MEDIA (ProFarmer + Others)

**Folder:** `trigger/ProFarmer/`

#### ProFarmer (CRITICAL - PAID ~$500/mo)

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| ProFarmer Login | `https://www.profarmer.com` | Auth | CRITICAL | ✅ Active |
| First Thing Today | `https://www.profarmer.com/news/first-thing-today` | Pre-market | CRITICAL | ✅ Scraping |
| Ahead of the Open | `https://www.profarmer.com/news/ahead-of-the-open` | Pre-market | CRITICAL | ✅ Scraping |
| After the Bell | `https://www.profarmer.com/news/after-the-bell` | Post-market | CRITICAL | ✅ Scraping |
| Agriculture News | `https://www.profarmer.com/news/agriculture-news` | Intraday | HIGH | ✅ Scraping |
| Market News | `https://www.profarmer.com/news/markets` | Intraday | HIGH | ✅ Scraping |
| Policy News | `https://www.profarmer.com/news/policy` | Intraday | HIGH | ✅ Scraping |
| Weather News | `https://www.profarmer.com/news/weather` | Intraday | HIGH | ✅ Scraping |
| Newsletters | `https://www.profarmer.com/newsletters` | Weekly | HIGH | ✅ Scraping |
| Weekly Outlook | `https://www.profarmer.com/newsletters/weekly-outlook` | Weekly | MEDIUM | ✅ Scraping |
| Grain Analysis | `https://www.profarmer.com/analysis/grains` | Analysis | HIGH | ✅ Scraping |
| Livestock Analysis | `https://www.profarmer.com/analysis/livestock` | Analysis | MEDIUM | ✅ Scraping |
| Energy Analysis | `https://www.profarmer.com/analysis/energy` | Analysis | MEDIUM | ✅ Scraping |
| Soybeans | `https://www.profarmer.com/markets/soybeans` | Commodity | CRITICAL | ✅ Scraping |
| Soybean Oil | `https://www.profarmer.com/markets/soybean-oil` | Commodity | CRITICAL | ✅ Scraping |
| Soybean Meal | `https://www.profarmer.com/markets/soybean-meal` | Commodity | CRITICAL | ✅ Scraping |
| Corn | `https://www.profarmer.com/markets/corn` | Commodity | HIGH | ✅ Scraping |
| Wheat | `https://www.profarmer.com/markets/wheat` | Commodity | MEDIUM | ✅ Scraping |
| Crude Oil | `https://www.profarmer.com/markets/crude-oil` | Commodity | HIGH | ✅ Scraping |
| Weather Forecast | `https://www.profarmer.com/weather/forecast` | Weather | HIGH | ✅ Scraping |
| Crop Conditions | `https://www.profarmer.com/weather/crop-conditions` | Weather | HIGH | ✅ Scraping |

**Trigger Job:** `trigger/ProFarmer/ProFarmerScripts/profarmer_all_urls.ts` ✅

#### Other Agricultural Media

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| Oil World | `https://www.oilworld.biz` | Global veg oil | CRITICAL | ⚠️ PAID |
| NOPA | `https://www.nopa.org` | Crush reports | CRITICAL | ⚠️ Needs job |
| AgriCensus | `https://www.agricensus.com` | Cash premiums | CRITICAL | ⚠️ PAID |
| Reuters Commodities | `https://www.reuters.com/markets/commodities` | Breaking news | CRITICAL | ⚠️ Needs job |
| DTN Progressive Farmer | `https://www.dtnpf.com/agriculture/web/ag/home` | U.S. farm news | HIGH | ⚠️ PAID |
| Soybean & Corn Advisor | `https://www.soybeansandcorn.com` | S. America crops | HIGH | ⚠️ Needs job |
| AgWeb Soybeans | `https://www.agweb.com/news/crops/soybeans` | U.S. soybeans | MEDIUM | ⚠️ Needs job |
| Farm Progress | `https://www.farmprogress.com/soybeans` | Field updates | MEDIUM | ⚠️ Needs job |
| Agriculture.com | `https://www.agriculture.com/markets-commodities` | Markets | MEDIUM | ⚠️ Needs job |
| Agrimoney Grains | `https://www.agrimoney.com/news/grains-oilseeds/` | Grain analysis | HIGH | ⚠️ Needs job |
| Agrimoney China | `https://www.agrimoney.com/news/china/` | China updates | CRITICAL | ⚠️ Needs job |
| World Grain | `https://www.world-grain.com/` | Global grain | MEDIUM | ⚠️ Needs job |

---

### 🎓 9. UNIVERSITY OF ILLINOIS FEEDS (FREE - MANDATORY)

**Folder:** `trigger/UofI_Feeds/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| Farm Policy News | `https://farmpolicynews.illinois.edu/` | Policy news | CRITICAL | ⚠️ Needs job |
| farmdoc Daily | `https://farmdocdaily.illinois.edu/` | Ag economics | CRITICAL | ⚠️ Needs job |
| farmdoc Main | `https://farmdoc.illinois.edu/` | Research | HIGH | - |

**Farm Policy News Categories:**
- `trade` → China, Tariff buckets
- `ethanol` → Biofuel bucket
- `budget` → Fed bucket
- `regulations` → Tariff, Biofuel buckets
- `immigration` → Policy risk

**farmdoc Daily Categories:**
- `biofuels/rins` → Biofuel bucket (Scott Irwin D4 RIN pricing)
- `agricultural-policy/trade` → China, Tariff buckets (Carl Zulauf)
- `marketing-and-outlook/grain-outlook` → Crush bucket (Nick Paulson)
- `finance/interest-rates` → Fed bucket (Michael Langemeier)
- `marketing-and-outlook/weekly-outlook` → All buckets

---

### 🏛️ 10. POLICY & THINK TANKS

**Folder:** `trigger/Policy/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| Federal Register API | `https://www.federalregister.gov/api/v1/documents.json` | Regulations | HIGH | ⚠️ Needs job |
| ICE News | `https://www.ice.gov/news/releases` | Immigration | MEDIUM | ⚠️ Needs job |
| DHS News | `https://www.dhs.gov/news-releases` | Security | MEDIUM | ⚠️ Needs job |
| CBP Newsroom | `https://www.cbp.gov/newsroom` | Border | MEDIUM | ⚠️ Needs job |
| FDACS Florida | `https://www.fdacs.gov/` | State ag | LOW | ⚠️ Needs job |
| Texas Agriculture | `https://www.texasagriculture.gov/` | State ag | LOW | ⚠️ Needs job |
| PIIE Trade War | `https://www.piie.com/research/piie-charts/us-china-trade-war-tariffs-date-chart` | Trade analysis | MEDIUM | ⚠️ Needs job |
| CSIS Trade Monitor | `https://www.csis.org/programs/scholl-chair-international-business/trade-war-monitor` | Trade intel | MEDIUM | ⚠️ Needs job |
| US-China Council | `https://www.uschina.org/` | Trade relations | MEDIUM | ⚠️ Needs job |
| Heritage Foundation | `https://www.heritage.org/agriculture` | Ag policy | MEDIUM | ⚠️ Needs job |
| America First Policy | `https://americafirstpolicy.com/` | Policy | MEDIUM | ⚠️ Needs job |
| Tax Foundation | `https://taxfoundation.org/research/all/federal/trade/` | Trade policy | MEDIUM | ⚠️ Needs job |
| AEI Trade Policy | `https://www.aei.org/tag/trade-policy/` | Trade policy | MEDIUM | ⚠️ Needs job |
| American Immigration Council | `https://immigrationimpact.com/` | Immigration | LOW | ⚠️ Needs job |
| Migration Policy Institute | `https://www.migrationpolicy.org/` | Immigration | LOW | ⚠️ Needs job |
| SPLC Immigrant Justice | `https://www.splcenter.org/issues/immigrant-justice` | Immigration | LOW | ⚠️ Needs job |

#### Farm & Labor Organizations

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| Farm Labor Organizing | `https://www.farmlabororganizing.org/` | Labor | LOW | ⚠️ Needs job |
| UFW | `https://ufw.org/` | Labor | LOW | ⚠️ Needs job |
| WGA | `https://www.wga.com/` | Industry | LOW | ⚠️ Needs job |
| Farm Bureau | `https://www.fb.org/newsroom/` | Industry | MEDIUM | ⚠️ Needs job |
| CA Farm Bureau | `https://www.cfbf.com/news/` | State | LOW | ⚠️ Needs job |
| GA Farm Bureau | `https://www.gfb.org/` | State | LOW | ⚠️ Needs job |
| Clean Fuels Alliance | `https://cleanfuels.org` | Biofuel lobby | MEDIUM | ⚠️ Needs job |

---

### 👤 11. ANALYSTS & SOCIAL (via ScrapeCreators)

**Folder:** `trigger/Analysts/`

| Analyst | Handle | Focus | Bucket | Priority | Status |
|---------|--------|-------|--------|----------|--------|
| Karen Braun | @kannbwx | USDA data, exports | Market Fundamentals | HIGH | ⚠️ Needs job |
| Arlan Suderman | @ArlanFF101 | Weather, S. America | Supply Weather | HIGH | ⚠️ Needs job |
| Scott Irwin | @ScottIrwinUIUC | RIN pricing | Biofuels | MEDIUM | ⚠️ Needs job |
| Michael Cordonnier | @SoybeanCorn | S. America crops | Supply Weather | HIGH | ⚠️ Needs job |
| Javier Blas | @JavierBlas | Energy, shipping | Logistics | MEDIUM | ⚠️ Needs job |

**Social Feed:**

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| Trump Truth Social | `https://truthsocial.com/@realDonaldTrump` | Policy signals | HIGH | ✅ Active |

---

### 🏝️ 12. VEGAS INTEL

**Folder:** `trigger/Vegas/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| Glide API | `https://api.glide.app/api/v1/` | Events data | MEDIUM | ⚠️ Needs job |
| Vegas Eater | `https://vegas.eater.com/` | Restaurant intel | LOW | ⚠️ Needs job |

---

### 🌍 13. TRADING ECONOMICS (PAID)

**Folder:** `trigger/TradingEconomics/`

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| TradingEconomics API | `https://api.tradingeconomics.com/` | Global data | HIGH | ⚠️ Needs key |
| TradingEconomics Docs | `https://docs.tradingeconomics.com/` | Reference | - | - |

**Coverage:** Global commodity prices, economic calendars, central bank rates, trade flow data

---

### 🌎 14. INTERNATIONAL SOURCES

#### Brazil

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| Conab | `https://www.conab.gov.br/ultimas-noticias` | Crop reports | HIGH | ⚠️ Needs job |
| ABIOVE | `https://abiove.org.br/en/statistics/` | Crush stats | HIGH | ⚠️ Needs job |
| Brazil Central Bank | `https://www3.bcb.gov.br/sgspub/` | FX data | MEDIUM | ⚠️ Needs job |

#### China

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| PBoC | `http://www.pbc.gov.cn/en/` | Monetary policy | MEDIUM | ⚠️ Needs job |

#### Europe

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| ECB SDW | `https://sdw-wsrest.ecb.europa.eu/service/` | EU data | MEDIUM | ⚠️ Needs job |

---

### 💰 15. FINANCIAL DATA APIS

| Source | URL | Data Type | Priority | Status |
|--------|-----|-----------|----------|--------|
| Treasury Fiscal Data | `https://api.fiscaldata.treasury.gov/services/api/v1/` | Treasury | MEDIUM | ⚠️ Needs job |
| BLS API | `https://api.bls.gov/publicAPI/v2/` | Labor stats | MEDIUM | ⚠️ Needs job |
| Fed Reserve Speeches | `https://www.federalreserve.gov/newevents/speech/` | Fed intel | MEDIUM | ⚠️ Needs job |
| Fed FOMC Calendar | `https://www.federalreserve.gov/monetarypolicy/fomccalendar.htm` | Events | HIGH | ⚠️ Needs job |

---

## 📊 BUCKET MAPPING TO BIG 8

| News Bucket | Big 8 Buckets | Key Sources |
|-------------|---------------|-------------|
| **Market Fundamentals** | Crush, China | Oil World, NOPA, AgriCensus, Reuters, ProFarmer |
| **Supply Weather** | Crush, China | Soybean & Corn Advisor, DTN, NOAA, INMET |
| **Demand Biofuels** | Biofuel, Energy | EPA RINs, EIA, Clean Fuels Alliance, farmdoc |
| **Trade Geo** | Tariff, China | Farm Policy News, Trump, PIIE, CSIS |
| **Logistics** | Crush, Energy | World Grain, Javier Blas |
| **Macro FX** | Fed, FX, Volatility | FRED (not news-based) |
| **Positioning** | Volatility | CFTC COT (not news-based) |

---

## ✅ IMPLEMENTATION STATUS SUMMARY

### Currently Active (7)
1. ✅ **DataBento** - 38 futures symbols
2. ✅ **FRED** - 24 macro indicators
3. ✅ **ProFarmer** - 22+ URLs, 3x daily
4. ✅ **ScrapeCreators** - 4 news buckets
5. ✅ **Trump Truth Social** - Hourly
6. ✅ **CFTC COT** - Script exists, needs Trigger job
7. ✅ **EIA** - API active

### Critical Priority (8) - Need Trigger Jobs
1. ⚠️ **EPA RIN Prices** - FREE, CRITICAL
2. ⚠️ **Farm Policy News** - FREE, CRITICAL
3. ⚠️ **farmdoc Daily** - FREE, CRITICAL
4. ⚠️ **Reuters Commodities** - CRITICAL
5. ⚠️ **Agrimoney China** - CRITICAL
6. ⚠️ **USDA Export Sales** - CRITICAL
7. ⚠️ **USDA WASDE** - CRITICAL
8. ⚠️ **Weather (NOAA/INMET)** - HIGH

### Paid Sources Needed (4)
1. 💰 **Oil World** - ~$2000/year
2. 💰 **AgriCensus** - ~$1500/year
3. 💰 **DTN** - ~$500/month
4. 💰 **TradingEconomics** - ~$200/month

---

## 📁 FOLDER STRUCTURE

```
trigger/
├── Analysts/
│   ├── Guides/
│   └── Scripts/
├── CFTC/
│   ├── Guides/
│   │   └── CFTC_COT_INGESTION.md
│   └── Scripts/
├── DataBento/
│   ├── Guides/
│   └── Scripts/
├── EIA_EPA/
│   ├── Guides/
│   └── Scripts/
├── FRED/
│   ├── Guides/
│   └── Scripts/
├── Policy/
│   ├── Guides/
│   └── Scripts/
├── ProFarmer/
│   ├── Guides/
│   └── Scripts/
│       └── profarmer_all_urls.ts ✅
├── ScrapeCreators/
│   ├── Guides/
│   │   └── NEWS_PIPELINE.md
│   └── Scripts/
├── TradingEconomics/
│   ├── Guides/
│   └── Scripts/
├── UofI_Feeds/
│   ├── Guides/
│   └── Scripts/
├── USDA/
│   ├── Guides/
│   └── Scripts/
├── Vegas/
│   ├── Guides/
│   └── Scripts/
├── Weather/
│   ├── Guides/
│   └── Scripts/
├── README.md
├── TRIGGER_SETUP.md
├── WEB_SCRAPING_TARGETS_MASTER.md
├── trigger.config.ts
└── Scripts/ (cross-source orchestration)
```

---

**Last Updated:** December 10, 2025
