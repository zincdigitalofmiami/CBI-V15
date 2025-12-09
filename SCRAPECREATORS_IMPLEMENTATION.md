# ScrapeCreators Implementation Guide

**Date:** December 9, 2024  
**Status:** ✅ Core infrastructure ready, credentials needed

---

## WHAT'S IMPLEMENTED

### ✅ ScrapeCreators API Integration

**API Key:** Already in `.env` as `SCRAPECREATORS_API_KEY=B1TOgQvMVSV6TDglqB8lJ2cirqi2`

**API Structure** (from scrapecreators.com):
- **Header:** `x-api-key: {your-api-key}`
- **Base URL:** `https://api.scrapecreators.com/v1/`
- **Pricing:** 1 request = 1 credit (pay as you go, credits never expire)

**Available Endpoints:**
1. **Google Search API** - `GET /google/search`
   - Params: `q` (query), `num` (results), `tbm=nws` (news search)
   - Use for: News aggregation, keyword-based scraping
   
2. **Truth Social API** - `GET /truthsocial/profile`
   - Params: `handle` (username)
   - Use for: Trump posts (already have collector for this)

3. **Twitter/X API** - `GET /twitter/profile`
   - Params: `handle` (username)
   - Use for: Ag influencers (Karen Braun, Arlan Suderman, etc.)

4. **Facebook API** - `GET /facebook/profile`
   - Use for: Farm organizations, think tanks

5. **Reddit API** - `GET /reddit/subreddit/posts`
   - Use for: r/farming, r/agriculture sentiment

---

## BUCKET COLLECTORS STATUS

| Bucket | File | API Used | Status |
|--------|------|----------|--------|
| **Biofuel Policy** | `collect_biofuel_policy.py` | ✅ Google Search | READY |
| **China Demand** | `collect_china_demand.py` | ⚠️ Needs update | 5 min |
| **Tariffs/Trade** | `collect_tariffs_trade_policy.py` | ⚠️ Needs update | 5 min |
| **Trump Truth Social** | `collect_trump_truth_social.py` | ⚠️ Needs Truth Social API | 5 min |

---

## DIRECT URL SCRAPING (Premium Sources)

**New File:** `src/ingestion/scrapecreators/direct_url_scraper.py`

**Purpose:** Scrape paywalled sources with authentication (DTN, Jacobsen, ProFarmer)

**Credentials Needed** (add to `.env`):
```bash
DTN_USERNAME=your_dtn_username
DTN_PASSWORD=your_dtn_password
JACOBSEN_USERNAME=your_jacobsen_email
JACOBSEN_PASSWORD=your_jacobsen_password
PROFARMER_USERNAME=your_profarmer_username
PROFARMER_PASSWORD=your_profarmer_password
```

**Sources to Scrape Daily:**
1. **DTN/Progressive Farmer** - Weather, logistics, basis, barge rates
2. **The Jacobsen** - RIN prices (D4/D6), biodiesel production, veg oil markets
3. **ProFarmer** - Market analysis, crop conditions

---

## CONFIGURATION FILES

### `config/ingestion/scrapecreators_targets.yaml`
Contains all target sources from V14:
- 5 Aggregation News Sites (AgriCensus, Reuters, Bloomberg, Farm Policy News, DTN)
- 5 Soybean Market Sites (Jacobsen, Oil World, Soybean & Corn Advisor, NOPA, USDA FAS)
- 5 People to Follow (Twitter/X handles for ag influencers)
- 5 Institutions (ADM, Bunge, Clean Fuels Alliance, Conab, EPA)
- 5 Policy/Laws to Watch (RFS, 45Z, LCFS, China Tariffs, UCO Import Bans)

---

## NEXT STEPS TO COMPLETE

### 1. Update Remaining Bucket Collectors (15 min)

**China Demand** (`collect_china_demand.py`):
```python
search_queries = [
    "China soybean imports demand",
    "Sinograin COFCO soybean purchases",
    "China crushing margins soybean",
    "China vegetable oil stocks"
]
```

**Tariffs/Trade** (`collect_tariffs_trade_policy.py`):
```python
search_queries = [
    "Section 301 tariffs soybeans",
    "USTR trade policy agriculture",
    "China retaliation tariffs",
    "WTO agriculture dispute"
]
```

**Trump Truth Social** (`collect_trump_truth_social.py`):
```python
# Use ScrapeCreators Truth Social API
endpoint = "https://api.scrapecreators.com/v1/truthsocial/profile"
params = {"handle": "realDonaldTrump"}
```

### 2. Add Missing Buckets (30 min each)

**Weather/Supply** - Create `collect_weather_supply.py`:
- Keywords: drought, La Niña, harvest, yield, planting, crop conditions

**Energy Complex** - Create `collect_energy_complex.py`:
- Keywords: crude oil, diesel, crack spread, HOBO spread, gasoline

**Macro Risk** - Create `collect_macro_risk.py`:
- Keywords: Fed, interest rates, VIX, financial stress, dollar strength

### 3. Add Premium Source Credentials

Fill in `.env` with your DTN, Jacobsen, ProFarmer logins.

### 4. Test End-to-End

```bash
# Test single bucket
python src/ingestion/scrapecreators/buckets/collect_biofuel_policy.py

# Test full pipeline
python src/ingestion/scrapecreators/collect_news_buckets.py

# Test direct URL scraper
python src/ingestion/scrapecreators/direct_url_scraper.py
```

### 5. Verify MotherDuck

```sql
SELECT 
    theme_primary,
    COUNT(*) as articles,
    MIN(date) as earliest,
    MAX(date) as latest
FROM raw.scrapecreators_news_buckets
GROUP BY theme_primary;
```

---

## SCRAPECREATORS API EXAMPLES

### Google News Search
```bash
curl "https://api.scrapecreators.com/v1/google/search?q=EPA+RFS&num=20&tbm=nws" \
  -H "x-api-key: B1TOgQvMVSV6TDglqB8lJ2cirqi2"
```

### Truth Social Profile
```bash
curl "https://api.scrapecreators.com/v1/truthsocial/profile?handle=realDonaldTrump" \
  -H "x-api-key: B1TOgQvMVSV6TDglqB8lJ2cirqi2"
```

### Twitter Profile
```bash
curl "https://api.scrapecreators.com/v1/twitter/profile?handle=kannbwx" \
  -H "x-api-key: B1TOgQvMVSV6TDglqB8lJ2cirqi2"
```

---

---

## ✅ IMPLEMENTATION COMPLETE

### What's Working Now:

**4 Bucket Collectors - READY TO RUN:**
1. ✅ `collect_biofuel_policy.py` - EPA, RFS, biodiesel, RIN, LCFS, 45Z
2. ✅ `collect_china_demand.py` - China imports, Sinograin, COFCO, crushing margins
3. ✅ `collect_tariffs_trade_policy.py` - Section 301, USTR, trade war, WTO
4. ✅ `collect_trump_truth_social.py` - Trump Truth Social posts (filtered for ag/trade)

**Smart Features:**
- ✅ **URL-based deduplication** - Keeps highest trust score version
- ✅ **Metadata enrichment** - Auto-tags impact (HIGH/MEDIUM/LOW) and horizon (FLASH/TACTICAL/STRUCTURAL)
- ✅ **Keyword filtering** - Trump posts filtered for ag/trade relevance
- ✅ **Search query tracking** - Tracks which query found each article (for tuning)
- ✅ **Source trust scores** - Premium sources (0.95), News aggregators (0.80)

**Database Schema:**
- ✅ Full 3-way segmentation (Thematic + Time-Horizon + Impact/Sentiment)
- ✅ 15 columns including url, search_query for debugging
- ✅ Compatible with MotherDuck `INSERT OR IGNORE` for idempotency

### Test It:

```bash
# Load environment
export $(grep -v '^#' .env | xargs)

# Test individual collectors
python scripts/test_scrapecreators_pipeline.py

# Run full pipeline
python src/ingestion/scrapecreators/collect_news_buckets.py
```

### Verify in MotherDuck:

```sql
-- Check ingestion
SELECT
    theme_primary,
    horizon,
    impact_magnitude,
    COUNT(*) as articles,
    MIN(date) as earliest,
    MAX(date) as latest
FROM raw.scrapecreators_news_buckets
GROUP BY theme_primary, horizon, impact_magnitude
ORDER BY articles DESC;

-- Check Trump posts
SELECT
    date,
    headline,
    zl_sentiment,
    impact_magnitude
FROM raw.scrapecreators_news_buckets
WHERE is_trump_related = true
ORDER BY date DESC
LIMIT 10;

-- Check deduplication
SELECT
    url,
    COUNT(*) as count
FROM raw.scrapecreators_news_buckets
WHERE url IS NOT NULL
GROUP BY url
HAVING COUNT(*) > 1;  -- Should return 0 rows
```

---

**Last Updated:** December 9, 2024 19:15 UTC

