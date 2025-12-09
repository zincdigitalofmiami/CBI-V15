# API Keys & Authentication Setup

## 🔑 Required API Keys

This document lists all API keys needed for CBI-V15 data ingestion.

---

## ✅ **Critical (Required for Core Functionality)**

### **1. MotherDuck**
**Purpose:** Cloud DuckDB database  
**Get it:** https://motherduck.com/  
**Env var:** `MOTHERDUCK_TOKEN`  
**Cost:** Free tier available

```bash
export MOTHERDUCK_TOKEN="your_token_here"
```

---

### **2. Databento**
**Purpose:** Market data (33 futures symbols)  
**Get it:** https://databento.com/  
**Env var:** `DATABENTO_API_KEY`  
**Cost:** Pay-as-you-go

```bash
export DATABENTO_API_KEY="your_key_here"
```

---

## 🔶 **Important (Recommended for Full Functionality)**

### **3. FRED (Federal Reserve Economic Data)**
**Purpose:** Macro indicators (24 series)  
**Get it:** https://fred.stlouisfed.org/docs/api/api_key.html  
**Env var:** `FRED_API_KEY`  
**Cost:** Free

**Series used:**
- FEDFUNDS, DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS7, DGS10, DGS20, DGS30
- T10Y2Y, T10Y3M, TEDRATE, NFCI, STLFSI4
- UNRATE, CPIAUCSL, GDP, PAYEMS
- VIXCLS, DTWEXBGS, DTWEXAFEGS, DTWEXEMEGS, PPOILUSDM

```bash
export FRED_API_KEY="your_key_here"
```

---

### **4. EIA (Energy Information Administration)**
**Purpose:** Biofuel data (RINs, biodiesel production)  
**Get it:** https://www.eia.gov/opendata/register.php  
**Env var:** `EIA_API_KEY`  
**Cost:** Free

**Endpoints used:**
- `/v2/crude-oil-imports/data/`
- `/v2/international/data/`
- `/v2/biofuels/biodiesel/production/`

```bash
export EIA_API_KEY="your_key_here"
```

---

### **5. NOAA CDO (Climate Data Online)**
**Purpose:** Weather data (14 agricultural regions)  
**Get it:** https://www.ncdc.noaa.gov/cdo-web/token  
**Env var:** `NOAA_API_TOKEN`  
**Cost:** Free

**Regions:**
- Brazil: Mato Grosso, Goiás, Mato Grosso do Sul, Paraná, Rio Grande do Sul, Bahia
- Argentina: Buenos Aires, Córdoba, Santa Fe, Entre Ríos
- United States: Eastern Corn Belt, Western Corn Belt, Northern Plains, Central Plains

```bash
export NOAA_API_TOKEN="your_token_here"
```

---

### **6. ScrapeCreators**
**Purpose:** Truth Social & Facebook sentiment data  
**Get it:** https://scrapecreators.com/  
**Env var:** `SCRAPECREATORS_API_KEY`  
**Cost:** Paid

**Endpoints:**
- `/v1/truthsocial` - Trump Truth Social posts
- `/v1/truthsocial/post` - Individual post
- `/v1/facebook/post` - Facebook posts

```bash
export SCRAPECREATORS_API_KEY="your_key_here"
```

---

## 🔷 **Optional (Nice to Have)**

### **7. USDA NASS QuickStats**
**Purpose:** U.S. agricultural statistics  
**Get it:** https://quickstats.nass.usda.gov/api  
**Env var:** `USDA_NASS_API_KEY`  
**Cost:** Free

```bash
export USDA_NASS_API_KEY="your_key_here"
```

---

### **8. Polygon.io**
**Purpose:** Alternative market data source  
**Get it:** https://polygon.io/  
**Env var:** `POLYGON_API_KEY`  
**Cost:** Free tier available

```bash
export POLYGON_API_KEY="your_key_here"
```

---

### **9. Copernicus (ECMWF)**
**Purpose:** Advanced weather data  
**Get it:** https://cds.climate.copernicus.eu/  
**Env var:** `COPERNICUS_API_KEY`  
**Cost:** Free

```bash
export COPERNICUS_API_KEY="your_key_here"
```

---

### **10. MarineTraffic**
**Purpose:** Shipping & port data  
**Get it:** https://www.marinetraffic.com/en/ais-api-services  
**Env var:** `MARINETRAFFIC_API_KEY`  
**Cost:** Paid

```bash
export MARINETRAFFIC_API_KEY="your_key_here"
```

---

## 🔧 **Setup Instructions**

### **1. Create `.env` file:**
```bash
cp config/env-templates/.env.example .env
```

### **2. Add your keys to `.env`:**
```bash
# Critical
MOTHERDUCK_TOKEN=your_token_here
DATABENTO_API_KEY=your_key_here

# Important
FRED_API_KEY=your_key_here
EIA_API_KEY=your_key_here
NOAA_API_TOKEN=your_token_here
SCRAPECREATORS_API_KEY=your_key_here

# Optional
USDA_NASS_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
COPERNICUS_API_KEY=your_key_here
MARINETRAFFIC_API_KEY=your_key_here
```

### **3. Load environment variables:**
```bash
source .env
```

Or use `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## ⚠️ **Security Best Practices**

### **DO:**
- ✅ Store keys in `.env` file (gitignored)
- ✅ Use environment variables
- ✅ Rotate keys regularly
- ✅ Use different keys for dev/prod
- ✅ Use GCP Secret Manager for production

### **DON'T:**
- ❌ Commit keys to git
- ❌ Hardcode keys in source code
- ❌ Share keys in Slack/email
- ❌ Use production keys in development

---

## 🔍 **Verify Keys**

```bash
# Verify all keys
python scripts/verify_api_keys.py

# Verify specific key
python scripts/verify_api_keys.py --key FRED_API_KEY
```

---

## 📚 **API Documentation**

- **Databento:** https://docs.databento.com/
- **FRED:** https://fred.stlouisfed.org/docs/api/
- **EIA:** https://www.eia.gov/opendata/documentation.php
- **NOAA CDO:** https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- **USDA NASS:** https://quickstats.nass.usda.gov/api
- **ScrapeCreators:** Contact support for docs

---

## ✅ **Summary**

**Minimum required:**
- MOTHERDUCK_TOKEN
- DATABENTO_API_KEY

**Recommended:**
- FRED_API_KEY
- EIA_API_KEY
- NOAA_API_TOKEN
- SCRAPECREATORS_API_KEY

**Total APIs: 10**  
**Free APIs: 7**  
**Paid APIs: 3** (Databento, ScrapeCreators, MarineTraffic)

