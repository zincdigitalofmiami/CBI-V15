# ScrapeCreators News Bucket System - Status

**Date:** December 9, 2024  
**Status:** ✅ Restored from archive, converting BigQuery → DuckDB

---

## WHAT WE PULLED FROM ARCHIVE

### ✅ Core Pipeline (`src/ingestion/scrapecreators/`)
- `collect_news_buckets.py` - Main orchestrator (converted to DuckDB)
- `sentiment_calculator.py` - FinBERT sentiment + zero-shot classification

### ✅ Bucket Collectors (`src/ingestion/scrapecreators/buckets/`)
| Bucket | File | Status | Big 8 Mapping |
|--------|------|--------|---------------|
| **Biofuel Policy** | `collect_biofuel_policy.py` | ✅ Updated with API call | Bucket 6 (Biofuel) |
| **China Demand** | `collect_china_demand.py` | ⚠️ Needs API implementation | Bucket 2 (China) |
| **Tariffs/Trade** | `collect_tariffs_trade_policy.py` | ⚠️ Needs API implementation | Bucket 5 (Tariff) |
| **Trump Truth Social** | `collect_trump_truth_social.py` | ⚠️ Needs API implementation | Bucket 5 (Tariff) |

### ✅ Configuration
- `config/ingestion/scrapecreators_targets.yaml` - Source definitions from V14

---

## BIG 8 BUCKET MAPPING

From `_archive/archive_legacy_20251204/docs/architecture/bucket_system.md`:

| Big 8 Bucket | News Bucket(s) | Dashboard Page |
|--------------|----------------|----------------|
| 1. Crush | (not news-driven) | Strategy |
| 2. China | China Demand | Trade Intelligence |
| 3. FX | (macro news) | Dashboard |
| 4. Fed | (macro news) | Dashboard |
| 5. Tariff | Trump Truth Social, Tariffs/Trade Policy | Trade Intelligence |
| 6. Biofuel | Biofuel Policy | Dashboard + Strategy |
| 7. Energy | (energy news) | Strategy |
| 8. Volatility | (macro risk news) | Sentiment |

**MISSING NEWS BUCKETS** (need to add):
- Weather/Supply (Big 8 implicit in fundamentals)
- Palm Substitution (Big 8 implicit in crush/energy)
- Energy Complex (Big 8 #7)
- Macro Risk (Big 8 #4, #8)
- Positioning (Big 8 implicit in CFTC data)

---

## NEWS BUCKET SCHEMA (from archive docs)

### `raw.scrapecreators_news_buckets`
```sql
CREATE TABLE raw.scrapecreators_news_buckets (
    date DATE,
    article_id TEXT PRIMARY KEY,
    
    -- Thematic Bucket (PRIMARY)
    theme_primary TEXT,  -- SUPPLY_WEATHER, DEMAND_BIOFUELS, TRADE_GEO, MACRO_FX, LOGISTICS, POSITIONING, IDIOSYNCRATIC
    
    -- Trump-Specific Tags
    is_trump_related BOOLEAN,
    policy_axis TEXT,  -- TRADE_CHINA, TRADE_TARIFFS, BIOFUELS_RFS, EPA_REGS, AGRICULTURE_SUBSIDY, GEOPOLITICS_SOY_ROUTE
    
    -- Time-Horizon Bucket
    horizon TEXT,  -- FLASH, TACTICAL, STRUCTURAL
    
    -- Impact & Sentiment (ZL-specific)
    zl_sentiment TEXT,  -- BULLISH_ZL, BEARISH_ZL, NEUTRAL
    impact_magnitude TEXT,  -- HIGH, MEDIUM, LOW
    sentiment_confidence FLOAT,
    sentiment_raw_score FLOAT,
    
    -- Raw Content
    headline TEXT,
    content TEXT,
    source TEXT,
    source_trust_score FLOAT,
    
    -- Metadata
    created_at TIMESTAMP
);
```

---

## NEXT STEPS

### 1. Complete Bucket Collectors (15 min each)
Update these 3 files with actual API calls (same pattern as `collect_biofuel_policy.py`):
- `collect_china_demand.py` - Keywords: China, import, export, trade, soybean, Sinograin, COFCO
- `collect_tariffs_trade_policy.py` - Keywords: tariff, trade war, USTR, Section 301, WTO
- `collect_trump_truth_social.py` - Use Truth Social API endpoint

### 2. Add Missing Buckets (30 min)
Create new collectors for:
- `collect_weather_supply.py` - Weather/harvest/yield news
- `collect_energy_complex.py` - Crude, diesel, crack spreads
- `collect_macro_risk.py` - Fed, VIX, financial stress

### 3. Update Database Schema (5 min)
Current `database/definitions/01_raw/scrapecreators_buckets.sql` is too simple.
Need to add all columns from archive schema above.

### 4. Test End-to-End (10 min)
```bash
python src/ingestion/scrapecreators/collect_news_buckets.py
```

### 5. Verify MotherDuck (2 min)
```sql
SELECT bucket_name, COUNT(*), MIN(date), MAX(date)
FROM raw.scrapecreators_news_buckets
GROUP BY bucket_name;
```

---

## API ENDPOINTS (from V14 reference)

ScrapeCreators API patterns:
- **News Search**: `https://api.scrapecreators.com/v1/news/search?keywords=...&days_back=7`
- **Truth Social**: `https://api.scrapecreators.com/v1/truthsocial?user=realDonaldTrump&days_back=7`
- **Facebook**: `https://api.scrapecreators.com/v1/facebook/post?page_id=...`

Auth: `Authorization: Bearer {SCRAPECREATORS_API_KEY}`

---

## SENTIMENT LOGIC (from archive)

**FinBERT** → Raw sentiment (positive/negative/neutral)  
**Zero-Shot Classification** → Bucket assignment  
**ZL Mapping Rules**:
- China buying = BULLISH_ZL
- EPA RFS increase = BULLISH_ZL
- Tariffs on soybeans = BEARISH_ZL (crushes demand)
- Crude up + biofuel mandate = BULLISH_ZL

---

**Last Updated:** December 9, 2024 18:20 UTC

