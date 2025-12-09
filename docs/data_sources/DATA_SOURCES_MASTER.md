# 📊 CBI-V15 Data Sources - Master Reference

## ⚠️ IMPORTANT: These Are Your ONLY Available Data Sources

**Do NOT reference or build features from sources not listed here.**

---

## 🌐 Foreign Exchange (FX Pairs)

### Spot FX (13 pairs)
- EUR/USD
- USD/JPY
- GBP/USD
- USD/CHF
- AUD/USD
- USD/CAD
- NZD/USD
- USD/CNY
- USD/BRL
- USD/MXN
- USD/ZAR
- USD/KRW
- USD/INR

**Source:** Databento (via CME FX futures conversion)  
**Prefix:** `fx_*`

---

## 📈 FX Futures (CME Globex)

### Available Contracts (10)
- **6E** – Euro FX
- **6J** – Japanese Yen
- **6B** – British Pound
- **6C** – Canadian Dollar
- **6A** – Australian Dollar
- **6N** – New Zealand Dollar
- **6M** – Mexican Peso
- **6L** – Brazilian Real
- **6S** – Swiss Franc
- **DX** – U.S. Dollar Index

**Source:** Databento CME Standard Feed
**Prefix:** `databento_*`
**Data:** OHLCV + tick data

---

## 💱 FX Forwards (CME, if available)

### Emerging Market & Major Pairs
- **USD/BRL** – U.S. Dollar / Brazilian Real
- **USD/CNY** – U.S. Dollar / Chinese Yuan
- **USD/EUR** – U.S. Dollar / Euro (inverse of EUR/USD)
- **USD/JPY** – U.S. Dollar / Japanese Yen
- **USD/MXN** – U.S. Dollar / Mexican Peso
- **USD/ZAR** – U.S. Dollar / South African Rand
- **USD/KRW** – U.S. Dollar / Korean Won
- **USD/INR** – U.S. Dollar / Indian Rupee

**Source:** Databento CME Feed (if available) or derived from spot + futures
**Prefix:** `fx_fwd_*`
**Data:** Forward rates, implied yields
**Status:** ⚠️ Verify availability via Databento

**Note:** Some FX forwards may need to be derived from spot FX + interest rate differentials if not directly available.

---

## 🌾 Commodity Futures (CBOT, NYMEX, COMEX, Bursa)

### Agricultural/Softs/Grains (11)
- **ZL** – Soybean Oil (CBOT) ← **PRIMARY SYMBOL** (RIN-driven since 2020)
- **ZS** – Soybeans (CBOT)
- **ZM** – Soybean Meal (CBOT)
- **ZC** – Corn (CBOT)
- **ZW** – Wheat (CBOT)
- **ZO** – Oats (CBOT)
- **ZR** – Rough Rice (CBOT) ✅ **VERIFIED CME**
- **HE** – Lean Hogs (CME)
- **LE** – Live Cattle (CME) ← **INFLATION HEDGE** (inverse to feed costs)
- **GF** – Feeder Cattle (CME)
- **FCPO** – Crude Palm Oil (Bursa Malaysia) ← **CRITICAL FOR ZL** (world's largest veg oil)

**Note:** ~~OJ (Orange Juice)~~ is **NOT available** via CME (trades on ICE U.S.)

### Energy/Refined Products (4)
- **CL** – WTI Crude Oil (NYMEX)
- **HO** – Heating Oil / ULSD (NYMEX) ← **ULSD is HO** (no separate UL symbol)
- **RB** – RBOB Gasoline (NYMEX)
- **NG** – Natural Gas (NYMEX)
- **FCPO** – Crude Palm Oil (Bursa) ← (also energy substitute)

**Note:** ~~UL (ULSD)~~ does **NOT exist** as separate symbol; use **HO**

### Metals (5)
- **HG** – Copper (COMEX) ← **CHINA GREEN INFRASTRUCTURE PROXY** (structural break 2022)
- **GC** – Gold (COMEX)
- **SI** – Silver (COMEX)
- **PL** – Platinum (NYMEX)
- **PA** – Palladium (NYMEX)

**Note:** ~~AL (Aluminum)~~ is **NOT available** via CME (trades on LME)

### Treasuries/Rate Futures (3)
- **ZF** – 5-Year Treasury Note (CBOT)
- **ZN** – 10-Year Treasury Note (CBOT) ← **Use ZN** (TY is floor symbol, same contract)
- **ZB** – 30-Year Treasury Bond (CBOT)

**Note:** ~~TY~~ is historical floor symbol; **use ZN** for 10Y Treasury

**Source:** Databento CME/NYMEX/COMEX Standard Feed
**Prefix:** `databento_*`
**Data:** OHLCV + tick data

---

## 📊 Macroeconomic Indicators (FRED)

### Interest Rates & Yields
- `fred_FEDFUNDS` – Federal Funds Rate
- `fred_DGS1MO` – 1-Month Treasury
- `fred_DGS3MO` – 3-Month Treasury
- `fred_DGS6MO` – 6-Month Treasury
- `fred_DGS1` – 1-Year Treasury
- `fred_DGS2` – 2-Year Treasury
- `fred_DGS5` – 5-Year Treasury
- `fred_DGS7` – 7-Year Treasury
- `fred_DGS10` – 10-Year Treasury
- `fred_DGS20` – 20-Year Treasury
- `fred_DGS30` – 30-Year Treasury

### Yield Spreads
- `fred_T10Y2Y` – 10Y-2Y Spread
- `fred_T10Y3M` – 10Y-3M Spread
- `fred_TEDRATE` – TED Spread

### Financial Conditions
- `fred_NFCI` – National Financial Conditions Index
- `fred_STLFSI4` – St. Louis Fed Financial Stress Index

### Economic Indicators
- `fred_UNRATE` – Unemployment Rate
- `fred_CPIAUCSL` – CPI (All Urban Consumers)
- `fred_GDP` – Gross Domestic Product
- `fred_PAYEMS` – Nonfarm Payrolls

### Market Indicators
- `fred_VIXCLS` – VIX (Volatility Index)
- `fred_DTWEXBGS` – Dollar Index (Broad)
- `fred_DTWEXAFEGS` – Dollar Index (Advanced Foreign Economies)
- `fred_DTWEXEMEGS` – Dollar Index (Emerging Markets)

### Commodity Prices
- `fred_PPOILUSDM` – Crude Oil Price (WTI)

**Source:** FRED API  
**Prefix:** `fred_*`  
**Frequency:** Daily (some monthly)

---

## 🛢️ Price Basis Series

### Soybean Oil
- CBOT (ZL) – Futures price
- Brazil FOB Paranaguá – Export price
- Argentina FOB Rosario – Export price
- CIF Rotterdam (Europe) – Import price

### Soybeans
- Brazil FOB Paranaguá – Export price
- Argentina FOB Rosario – Export price

### Soybean Meal
- Argentina FOB Rosario – Export price

**Source:** USDA / Market data providers  
**Prefix:** `usda_*` or `basis_*`  
**Frequency:** Weekly/Daily

---

## 🌦️ Weather Regions (NOAA)

### Brazil (6 regions)
- Mato Grosso
- Goiás
- Mato Grosso do Sul
- Paraná
- Rio Grande do Sul
- Bahia

### Argentina (4 regions)
- Buenos Aires
- Córdoba
- Santa Fe
- Entre Ríos

### United States (4 regions)
- Eastern Corn Belt (IL, IN, OH)
- Western Corn Belt (IA, MN, NE)
- Northern Plains (ND, SD)
- Central Plains (KS, NE)

**Source:** NOAA GFS/GSOD  
**Prefix:** `weather_*`  
**Data:** Temperature, precipitation, drought indices

---

## 📣 Sentiment & News Buckets (ScrapeCreators)

### Available Buckets (8)
- `scrc_biofuel_policy` – RFS, biodiesel mandates
- `scrc_china_demand` – Import demand signals
- `scrc_tariffs_trade` – Trade policy, tariffs
- `scrc_us_politics` – Trump/Truth Social signals
- `scrc_market_volatility` – VIX, risk-off sentiment
- `scrc_crop_failures` – Weather, disease, pests
- `scrc_supply_chain` – Logistics, shipping
- `scrc_general_market` – General commodity news

**Source:** ScrapeCreators API  
**Prefix:** `scrc_*` or `policy_trump_*`  
**Data:** Sentiment scores, article counts, keyword frequency

---

## 🌱 USDA Public Series

### WASDE (World Agricultural Supply & Demand)
- `usda_wasde_world_soyoil_prod` – Global soy oil production
- `usda_wasde_world_soymeal_prod` – Global soy meal production
- `usda_wasde_world_soybeans_prod` – Global soybean production

### Export Sales
- `usda_export_soybeans_weekly` – Weekly export sales
- `usda_export_soyoil_weekly` – Weekly soy oil exports
- `usda_export_soymeal_weekly` – Weekly soy meal exports

### Crop Progress & Conditions
- `usda_crop_progress_soybeans` – % planted, emerged, etc.
- `usda_crop_conditions_soybeans` – Good/Excellent ratings
- `usda_crop_progress_corn` – % planted, emerged, etc.
- `usda_crop_conditions_corn` – Good/Excellent ratings

**Source:** USDA Open APIs  
**Prefix:** `usda_*`  
**Frequency:** Weekly/Monthly

---

## 🏛️ EIA (Biofuels & RINs)

### RIN Prices
- `eia_RIN_D4` – Biomass Diesel RINs
- `eia_RIN_D6` – Ethanol RINs

### Biofuel Production & Consumption
- `eia_BIODIESEL_PROD` – Monthly biodiesel production
- `eia_BIODIESEL_CONSUMPTION` – Monthly consumption
- `eia_RFS_VOLUMES` – Renewable Volume Obligations

### Petroleum Products
- `eia_ULSD_WHOLESALE_MIDWEST` – Ultra-low sulfur diesel prices

**Source:** EIA Open API
**Prefix:** `eia_*`
**Frequency:** Weekly/Monthly

---

## 📦 CFTC Commitment of Traders (COT)

### Available Reports (All Futures Symbols)

**Commodity Futures:**
- Agricultural: ZL, ZS, ZM, ZC, ZW, ZO, ZR, HE, LE, GF
- Energy: CL, HO, RB, NG
- Metals: HG, GC, SI, PL, PA
- Softs: OJ (if available)

**FX Futures:**
- 6E (Euro), 6J (Yen), 6B (Pound), 6C (CAD), 6A (AUD), 6N (NZD), 6M (MXN), 6L (BRL), 6S (CHF), DX (Dollar Index)

**Treasury Futures:**
- ZF (5Y), ZN (10Y), ZB (30Y), TY (10Y alt)

**Data Fields:**
- Net positions (commercial, non-commercial, managed money)
- Long positions
- Short positions
- Open interest
- Spreads
- Change from prior week

**Source:** CFTC Public Data (https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
**Prefix:** `cftc_*`
**Frequency:** Weekly (Tuesday 3:30 PM ET release, data as of prior Tuesday)
**Status:** ⚠️ Pending ingestion pipeline

**Example features:**
- `cftc_ZL_net_noncomm` – Net non-commercial positions (speculators)
- `cftc_ZL_net_comm` – Net commercial positions (hedgers)
- `cftc_ZL_managed_money_long` – Managed money long positions
- `cftc_ZL_open_interest` – Total open interest
- `cftc_ZL_net_noncomm_pct_oi` – Net non-comm as % of open interest

---

## ✅ Confirmed Infrastructure & Tool Access

### Data Platforms
- ✅ **DuckDB** (local database)
- ✅ **MotherDuck** (cloud database, token loaded)
- ✅ **Databento** (CME Standard Feed access)
- ✅ **FRED API** (Federal Reserve Economic Data)
- ✅ **NOAA** (GFS/GSOD weather data)
- ✅ **ScrapeCreators API** (sentiment & news)
- ✅ **USDA Open APIs** (WASDE, export, crop data)
- ✅ **EIA Open API** (biofuels & petroleum)

### Ingestion Scripts
- ✅ `src/ingestion/databento/ingest_daily.py`
- ✅ `src/ingestion/fred/ingest_daily.py`
- ✅ `src/ingestion/eia/ingest_biofuels.py`
- ✅ `src/ingestion/scrape_creator/ingest_trump_posts.py`
- ⚠️ `src/ingestion/noaa/ingest_weather.py` (pending)
- ⚠️ `src/ingestion/usda/ingest_wasde.py` (pending)
- ⚠️ `src/ingestion/cftc/ingest_cot.py` (pending)

---

## 📋 Feature Engineering Prefixes

**All features must use these prefixes:**

| Prefix | Source | Example |
|--------|--------|---------|
| `fred_*` | FRED economic data | `fred_FEDFUNDS`, `fred_DGS10` |
| `fx_*` | FX pair derived features | `fx_EURUSD_ret_1d`, `fx_USDJPY_volatility_21d` |
| `databento_*` | Futures OHLCV + tick | `databento_ZL_close`, `databento_CL_volume` |
| `weather_*` | NOAA regional weather | `weather_brazil_mato_grosso_precip` |
| `eia_*` | Biofuel & petroleum | `eia_RIN_D4`, `eia_BIODIESEL_PROD` |
| `usda_*` | WASDE, export, crop | `usda_export_soybeans_weekly` |
| `cftc_*` | COT data | `cftc_ZL_net_noncomm` |
| `scrc_*` | Sentiment & news | `scrc_biofuel_policy_sentiment` |
| `policy_trump_*` | Trump/Truth Social | `policy_trump_tariff_mentions` |
| `basis_*` | Price basis spreads | `basis_brazil_fob_paranagua` |

---

## 🚫 What We DON'T Have

**Do NOT build features from these sources (not available):**

- ❌ Bloomberg Terminal data
- ❌ Reuters data
- ❌ ICE futures (only CME/NYMEX/COMEX/Bursa)
- ❌ Options data (only futures)
- ❌ Order book depth beyond Databento tick data
- ❌ Alternative data (satellite, shipping, etc.)
- ❌ Social media beyond ScrapeCreators
- ❌ Proprietary research reports

---

## 📊 Data Coverage Summary

| Category | Symbols/Series | Source | Status |
|----------|----------------|--------|--------|
| **FX Spot Pairs** | 13 pairs | Databento/Derived | ✅ Available |
| **FX Futures** | 10 contracts | Databento CME | ✅ Available |
| **FX Forwards** | 8 pairs | Databento/Derived | ⚠️ Verify availability |
| **Commodity Futures** | 24 contracts | Databento CME/NYMEX/COMEX/Bursa | ✅ Available |
| **Treasuries** | 4 contracts | Databento CBOT | ✅ Available |
| **Macro Indicators** | 24 series | FRED | ✅ Available |
| **Weather** | 14 regions | NOAA | ⚠️ Pending ingestion |
| **Sentiment** | 8 buckets | ScrapeCreators | ✅ Available |
| **USDA** | 10 series | USDA APIs | ⚠️ Pending ingestion |
| **Biofuels** | 6 series | EIA | ✅ Available |
| **CFTC COT** | All futures | CFTC | ⚠️ Pending ingestion |
| **Price Basis** | 7 series | USDA/Market data | ⚠️ Pending ingestion |

**Total Futures Symbols: 38** (10 FX + 24 Commodities + 4 Treasuries)
**Total FX Coverage: 31 pairs** (13 spot + 10 futures + 8 forwards)

---

## 🎯 Primary Symbol: ZL (Soybean Oil)

**All models are built to forecast ZL.**

**Key relationships:**
- **Crush spread:** (ZM × 0.022 + ZL × 11) - ZS
- **BOHO spread:** (ZL/100 × 7.5) - HO (Soy Oil vs Heating Oil)
- **Biofuel demand:** eia_RIN_D4, eia_BIODIESEL_PROD
- **China demand:** HG (copper proxy), usda_export_soybeans_weekly
- **Energy substitution:** CL, HO (crude & heating oil)
- **FX effects:** DX (dollar index), fx_USDBRL, fx_USDCNY

---

## 📝 Usage in Feature Engineering

**When building features, ONLY use data from this master list.**

**Example valid features:**
```python
# ✅ VALID - uses available data
fred_DGS10_lag_1d
databento_ZL_sma_21
eia_RIN_D4_momentum_10d
scrc_biofuel_policy_sentiment_7d
fx_USDBRL_ret_21d
weather_brazil_mato_grosso_precip_30d_avg
```

**Example invalid features:**
```python
# ❌ INVALID - data not available
bloomberg_ZL_implied_vol  # No Bloomberg access
ice_palm_oil_close        # No ICE futures
twitter_sentiment_zl      # No Twitter API (use ScrapeCreators)
satellite_brazil_ndvi     # No satellite data
```

---

## 🔄 Update Frequency

| Source | Frequency | Latency |
|--------|-----------|---------|
| Databento (futures) | Real-time | < 1 second |
| FRED | Daily | 1 day |
| EIA | Weekly/Monthly | 1-7 days |
| USDA | Weekly/Monthly | 1-7 days |
| ScrapeCreators | Daily | < 1 day |
| NOAA Weather | Daily | 1 day |
| CFTC COT | Weekly | 3 days (Friday data, Tuesday release) |

---

## ✅ Summary

**Total available data sources:**

### **FX Coverage (31 pairs total):**
- **13 FX spot pairs** (EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD, USD/CNY, USD/BRL, USD/MXN, USD/ZAR, USD/KRW, USD/INR)
- **10 FX futures** (6E, 6J, 6B, 6C, 6A, 6N, 6M, 6L, 6S, DX)
- **8 FX forwards** (USD/BRL, USD/CNY, USD/EUR, USD/JPY, USD/MXN, USD/ZAR, USD/KRW, USD/INR)

### **Commodity Futures (24 symbols):**
- **12 Agricultural/Softs** (ZL, ZS, ZM, ZC, ZW, ZO, ZR, OJ, HE, LE, GF, FCPO)
- **6 Energy** (CL, HO, RB, NG, UL, FCPO)
- **6 Metals** (HG, GC, SI, PL, PA, AL)

### **Treasuries/Rate Futures (4 symbols):**
- ZF (5Y), ZN (10Y), ZB (30Y), TY (10Y alt)

### **Macro & Fundamental Data:**
- **24 FRED macro indicators** (rates, yields, spreads, unemployment, CPI, GDP, VIX, dollar indices)
- **14 weather regions** (Brazil, Argentina, US Corn Belt)
- **8 sentiment buckets** (ScrapeCreators: biofuel, China, tariffs, Trump, volatility, crops, supply chain)
- **10 USDA series** (WASDE, export sales, crop progress/conditions)
- **6 EIA biofuel series** (RIN D4/D6, biodiesel production/consumption, RFS volumes, ULSD)
- **7 price basis series** (Brazil/Argentina FOB, Rotterdam CIF)
- **CFTC COT** (all futures - net positions, open interest, managed money)

**Total Futures Symbols: 38** (10 FX + 24 Commodities + 4 Treasuries)

**CRITICAL: FCPO (Palm Oil) is essential for ZL modeling** - palm oil is the world's largest vegetable oil and directly competes with soybean oil.

**This is your complete data universe. Do not reference sources outside this list.**

