# PHASE 1: CRITICAL DATA FEEDS

**Goal:** Implement missing data sources required for Big 8 bucket coverage  
**Status:** NOT STARTED  
**Dependencies:** Phase 0 complete  
**Estimated Time:** 12-16 hours  
**Risk Level:** MEDIUM (data availability dependent)

---

## 📋 TASKS (7 total)

### Task 1.1: Create EPA RIN Prices Trigger Job (CRITICAL)
**UUID:** `qhEr1yaxasB62K3v9fQxfD`

**Source:** EPA EMTS RIN Trades and Price Information (FREE)  
**URL:** https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information

**File:** `trigger/epa_rin_prices.ts`

**Data:** Weekly volume-weighted average RIN prices (D3, D4, D5, D6)  
**Target Table:** `raw.eia_petroleum` (series_id: rin_d3_price, rin_d4_price, rin_d5_price, rin_d6_price)  
**Frequency:** Weekly (updated monthly by EPA)

**Implementation:**
```typescript
import { task } from "@trigger.dev/sdk/v3";
import * as cheerio from "cheerio";

export const epaRinPrices = task({
  id: "epa-rin-prices",
  run: async (payload: { startDate?: string }) => {
    // Scrape EPA Qlik Sense reports
    const url = "https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information";
    const response = await fetch(url);
    const html = await response.text();
    const $ = cheerio.load(html);
    
    // Parse RIN prices (D3, D4, D5, D6)
    const rinPrices = parseRinPrices($);
    
    // Insert into raw.eia_petroleum with ON CONFLICT
    await insertRinPrices(rinPrices);
    
    return { rowsInserted: rinPrices.length };
  },
});
```

**Validation:**
```sql
SELECT COUNT(*) FROM raw.eia_petroleum 
WHERE series_id LIKE 'rin_%' AND date >= '2010-07-01';
-- Expected: >700 rows (weekly data from July 2010 to present)
```

**Big 8 Impact:** Biofuel (bucket 6)

---

### Task 1.2: Remove Mock Data from USDA Export Sales (HIGH RISK)
**UUID:** `jWcKBJg3duV76V8bBvENtZ`

**Problem:** Uses `generate_mock_export_sales()` - violates NO FAKE DATA rule

**File:** `src/ingestion/usda/ingest_export_sales.py`

**Changes:**
```python
# REMOVE:
def generate_mock_export_sales():
    # DELETE entire function

# ADD:
def fetch_real_usda_export_sales():
    """Fetch real USDA FAS export sales data."""
    api_url = "https://apps.fas.usda.gov/esrquery/api/v1/export-sales"
    params = {
        'commodity': 'SOYBEANS',
        'country': 'ALL',
        'startDate': '2020-01-01'
    }
    response = requests.get(api_url, params=params)
    return response.json()
```

**Create Trigger Job:** `trigger/usda_export_sales.ts`

**Validation:**
```sql
SELECT COUNT(*) FROM raw.usda_export_sales 
WHERE source != 'mock' AND commodity = 'SOYBEANS';
-- Expected: >500 rows of real weekly export sales data
```

**Big 8 Impact:** China (bucket 2)

---

### Task 1.3: Create CFTC COT Trigger Job (MEDIUM RISK)
**UUID:** `1WSHKdU8kH5Wm1px7ouRWV`

**Existing:** `src/ingestion/cftc/ingest_cot.py` (script exists, needs Trigger wrapper)

**File:** `trigger/cftc_cot_ingest.ts`

**Source:** CFTC Disaggregated COT Reports  
**Target Tables:** `raw.cftc_cot_disaggregated`, `raw.cftc_cot_tff`  
**Frequency:** Weekly (Friday 3:30 PM ET release, data as of prior Tuesday)

**Implementation:**
```typescript
export const cftcCotIngest = task({
  id: "cftc-cot-ingest",
  run: async () => {
    // Call existing Python script
    const result = await exec("python src/ingestion/cftc/ingest_cot.py");
    return result;
  },
});
```

**Validation:**
```sql
SELECT COUNT(DISTINCT symbol) FROM raw.cftc_cot_disaggregated 
WHERE report_date >= CURRENT_DATE - INTERVAL '30 days';
-- Expected: 38 symbols with recent data
```

**Big 8 Impact:** All buckets (positioning overlay)

---

### Task 1.4: Create FRED Daily Ingest Trigger Job (MEDIUM RISK)
**UUID:** `2op44NsCXwbBqB825ePgqy`

**Existing:** `trigger/fred_seed_harvest.ts` (discovery only, not daily ingest)

**File:** `trigger/fred_daily_ingest.ts`

**Purpose:** Daily ingestion of all active FRED series  
**Target Table:** `raw.fred_economic`  
**Series:** All series in `raw.fred_series_metadata WHERE is_active = true`  
**Frequency:** Daily at 10 AM UTC (after FRED updates)

**Implementation:**
```typescript
export const fredDailyIngest = task({
  id: "fred-daily-ingest",
  run: async () => {
    // Get active series from metadata
    const activeSeries = await getActiveFredSeries();
    
    // Fetch latest observations for each series
    for (const series of activeSeries) {
      await fetchFredSeries(series.series_id);
    }
    
    return { seriesUpdated: activeSeries.length };
  },
});
```

**Validation:**
```sql
SELECT COUNT(DISTINCT series_id) FROM raw.fred_economic 
WHERE date = CURRENT_DATE;
-- Expected: 24+ series updated daily
```

**Big 8 Impact:** Fed (bucket 4), FX (bucket 3), Volatility (bucket 8)

---

### Task 1.5: Create Farm Policy News Scraper (CRITICAL)
**UUID:** `bLrY677E6qa79dw2Ccuwrz`

**Source:** Farm Policy News (University of Illinois) - FREE, MANDATORY  
**URL:** https://farmpolicynews.illinois.edu/

**File:** `trigger/farm_policy_news.ts`

**Categories:** trade, ethanol, budget, regulations, immigration  
**Target Table:** `raw.bucket_news` (source: 'farm_policy_news')  
**Frequency:** Hourly check for new articles

**Why Critical:** Real-time China/tariff policy directly impacting soybeans

**Example Headlines:**
- "China Soybean Buying Deadline Now February"
- "$11B Bridge Farm Aid"

**Implementation:**
```typescript
export const farmPolicyNews = task({
  id: "farm-policy-news",
  run: async () => {
    const url = "https://farmpolicynews.illinois.edu/";
    const response = await fetch(url);
    const $ = cheerio.load(await response.text());
    
    const articles = [];
    $('.article').each((i, el) => {
      articles.push({
        title: $(el).find('.title').text(),
        category: $(el).find('.category').text(),
        url: $(el).find('a').attr('href'),
        published_at: $(el).find('.date').text(),
      });
    });
    
    // Insert into raw.bucket_news
    await insertBucketNews(articles, 'farm_policy_news');
    
    return { articlesFound: articles.length };
  },
});
```

**Validation:**
```sql
SELECT COUNT(*) FROM raw.bucket_news 
WHERE source = 'farm_policy_news' 
AND created_at >= CURRENT_DATE - INTERVAL '7 days';
-- Expected: 10-30 articles per week
```

**Big 8 Impact:** China (bucket 2), Tariff (bucket 5), Fed (bucket 4)

---

### Task 1.6: Create farmdoc Daily Scraper (HIGH RISK)
**UUID:** `3Chx3x3SceMm6z3UgwRxJM`

**Source:** farmdoc Daily (University of Illinois) - FREE, HIGH VALUE
**URL:** https://farmdocdaily.illinois.edu/

**File:** `trigger/farmdoc_daily.ts`

**Categories:** biofuels/rins (Scott Irwin), trade policy, grain outlook, interest rates, weekly outlook
**Target Table:** `raw.bucket_news` (source: 'farmdoc_daily')
**Frequency:** Daily at 6 AM UTC

**Why Critical:** Scott Irwin RIN pricing models (75% R² accuracy), trade policy analysis

**Big 8 Impact:** Biofuel (bucket 6), China (bucket 2), Tariff (bucket 5), Fed (bucket 4), Crush (bucket 1)

---

### Task 1.7: Validate Phase 1 Complete (CRITICAL)
**UUID:** `akuiGumfvyrV1dsW5vRKGD`

**Validation Commands:**
```sql
-- 1. EPA RIN prices
SELECT COUNT(*) FROM raw.eia_petroleum WHERE series_id LIKE 'rin_%';
-- Expected: >700 rows (weekly since 2010)

-- 2. USDA export sales (no mock data)
SELECT COUNT(*) FROM raw.usda_export_sales WHERE source != 'mock';
-- Expected: >500 rows

-- 3. CFTC COT coverage
SELECT COUNT(DISTINCT symbol) FROM raw.cftc_cot_disaggregated;
-- Expected: 38 symbols

-- 4. FRED active series
SELECT COUNT(DISTINCT series_id) FROM raw.fred_economic;
-- Expected: 24+ series

-- 5. News sources
SELECT COUNT(*) FROM raw.bucket_news
WHERE source IN ('farm_policy_news', 'farmdoc_daily');
-- Expected: 15-40 articles per week
```

**Success Criteria:**
- ✅ All Big 8 buckets have required data sources
- ✅ No mock/placeholder data remains
- ✅ Data freshness within expected intervals

**⚠️ STOP:** Phase 2-5 depend on complete data coverage

---

## 📊 PHASE 1 SUMMARY

| Metric | Value |
|--------|-------|
| **Total Tasks** | 7 |
| **Critical** | 2 tasks (EPA RIN, Farm Policy News) |
| **High Risk** | 2 tasks |
| **Medium Risk** | 3 tasks |
| **Estimated Time** | 12-16 hours |

**Big 8 Coverage Validation:**
- ✅ Crush: Databento (ZL/ZS/ZM), NOPA, farmdoc Grain Outlook
- ✅ China: Farm Policy News Trade, USDA Export Sales, farmdoc Trade
- ✅ FX: FRED FX series, Databento (6L/DX)
- ✅ Fed: FRED rates/curve, Farm Policy News Budget
- ✅ Tariff: Farm Policy News Trade, ScrapeCreators Trump
- ✅ Biofuel: EPA RIN Prices, EIA, farmdoc RINs
- ✅ Energy: EIA petroleum, Databento (CL/HO/RB)
- ✅ Volatility: FRED VIXCLS, Databento VIX, STLFSI4

**Next Phase:** Phase 2 (AutoGluon Integration)


