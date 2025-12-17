# FRED API Complete Reference
**Last Updated:** December 16, 2025  
**Purpose:** Complete documentation of FRED API endpoints for historical data backfill  
**API Docs:** https://fred.stlouisfed.org/docs/api/fred/

---

## Why This Matters

FRED (Federal Reserve Economic Data) provides **500,000+ economic time series** with complete historical data going back to the 1940s-1960s. This is our **primary source** for:
- Fed monetary policy (interest rates, fed funds, SOFR)
- Financial conditions (NFCI, STLFSI, stress indices)
- FX rates (all major currencies)
- Macro indicators (GDP, CPI, employment)
- Energy prices (crude oil, petroleum products)

**Key Advantage:** One API call can get ENTIRE historical series (no pagination limits for individual series).

---

## API Authentication

All requests require API key:
```
https://api.stlouisfed.org/fred/endpoint?api_key=YOUR_KEY&file_type=json
```

**Get API Key:** https://fred.stlouisfed.org/docs/api/api_key.html  
**Environment Variable:** `FRED_API_KEY`

---

## 1. Categories (Hierarchical Organization)

### `fred/category`
Get a category by ID.

**Example:**
```
GET /fred/category?category_id=125&api_key=YOUR_KEY
```

**Key Categories:**
- `32991` - Money, Banking, & Finance
- `32455` - Interest Rates
- `94` - Foreign Exchange Rates
- `32263` - Monetary Data
- `32348` - Financial Indicators

### `fred/category/children`
Get child categories for a parent category.

**Example:**
```
GET /fred/category/children?category_id=125&api_key=YOUR_KEY
```

**Use Case:** Discover all subcategories under "Interest Rates" to find all Fed policy series.

### `fred/category/related`
Get related categories.

**Example:**
```
GET /fred/category/related?category_id=125&api_key=YOUR_KEY
```

### `fred/category/series`
Get all series in a category.

**Example:**
```
GET /fred/category/series?category_id=32455&api_key=YOUR_KEY
```

**Use Case:** Get ALL interest rate series in one call.

### `fred/category/tags`
Get tags for a category.

**Example:**
```
GET /fred/category/tags?category_id=125&api_key=YOUR_KEY
```

### `fred/category/related_tags`
Get related tags for a category.

**Example:**
```
GET /fred/category/related_tags?category_id=125&tag_names=monetary+policy&api_key=YOUR_KEY
```

---

## 2. Releases (Complete Historical Data)

### `fred/releases` ⭐ **CRITICAL**
Get all releases of economic data.

**Example:**
```
GET /fred/releases?api_key=YOUR_KEY
```

**Returns:** List of ALL FRED releases with IDs.

**Key Releases:**
- `62` - H.15 Selected Interest Rates
- `469` - Financial Stress Index
- `456` - Chicago Fed NFCI
- `17` - H.10 Foreign Exchange Rates
- `52` - Z.1 Financial Accounts
- `50` - Employment Situation
- `10` - Consumer Price Index

### `fred/releases/dates`
Get release dates for all releases.

**Example:**
```
GET /fred/releases/dates?api_key=YOUR_KEY
```

**Use Case:** Find when new data is published to schedule daily pulls.

### `fred/release` ⭐ **CRITICAL**
Get details about a specific release.

**Example:**
```
GET /fred/release?release_id=62&api_key=YOUR_KEY
```

### `fred/release/dates`
Get release dates for a specific release.

**Example:**
```
GET /fred/release/dates?release_id=62&api_key=YOUR_KEY
```

### `fred/release/series`
Get all series in a release.

**Example:**
```
GET /fred/release/series?release_id=62&api_key=YOUR_KEY
```

**Use Case:** Discover all series in H.15 Interest Rates release.

### `fred/release/sources`
Get sources for a release.

**Example:**
```
GET /fred/release/sources?release_id=62&api_key=YOUR_KEY
```

### `fred/release/tags`
Get tags for a release.

**Example:**
```
GET /fred/release/tags?release_id=62&api_key=YOUR_KEY
```

### `fred/release/related_tags`
Get related tags for a release.

**Example:**
```
GET /fred/release/related_tags?release_id=62&tag_names=interest+rate&api_key=YOUR_KEY
```

### `fred/release/tables`
Get release tables (structured data).

**Example:**
```
GET /fred/release/tables?release_id=62&api_key=YOUR_KEY
```

---

## 3. Series (Individual Time Series)

### `fred/series` ⭐ **CRITICAL**
Get metadata for a series.

**Example:**
```
GET /fred/series?series_id=DFEDTARU&api_key=YOUR_KEY
```

**Returns:**
- Title, frequency, units
- Observation start/end dates
- Last updated timestamp
- Seasonal adjustment info

### `fred/series/categories`
Get categories for a series.

**Example:**
```
GET /fred/series/categories?series_id=DFEDTARU&api_key=YOUR_KEY
```

### `fred/series/observations` ⭐ **GOLD STANDARD**
Get actual data values for a series.

**Example:**
```
GET /fred/series/observations?series_id=DFEDTARU&api_key=YOUR_KEY
```

**Parameters:**
- `observation_start` - Start date (YYYY-MM-DD)
- `observation_end` - End date (YYYY-MM-DD)
- `limit` - Max observations (default 100,000)
- `sort_order` - asc or desc

**Use Case:** Get COMPLETE historical data for any series (e.g., Fed Funds rate from 1954-present).

### `fred/series/release`
Get the release for a series.

**Example:**
```
GET /fred/series/release?series_id=DFEDTARU&api_key=YOUR_KEY
```

### `fred/series/search` ⭐ **DISCOVERY**
Search for series by keywords.

**Example:**
```
GET /fred/series/search?search_text=soybean&api_key=YOUR_KEY
```

**Use Case:** Find all soybean-related series (prices, production, exports).

### `fred/series/search/tags`
Get tags for a series search.

**Example:**
```
GET /fred/series/search/tags?series_search_text=soybean&api_key=YOUR_KEY
```

### `fred/series/search/related_tags`
Get related tags for a series search.

**Example:**
```
GET /fred/series/search/related_tags?series_search_text=soybean&tag_names=agriculture&api_key=YOUR_KEY
```

### `fred/series/tags`
Get tags for a series.

**Example:**
```
GET /fred/series/tags?series_id=DFEDTARU&api_key=YOUR_KEY
```

### `fred/series/updates`
Get series sorted by when they were updated.

**Example:**
```
GET /fred/series/updates?api_key=YOUR_KEY
```

**Use Case:** Find which series have new data today (for daily refresh).

### `fred/series/vintagedates`
Get dates when a series was revised.

**Example:**
```
GET /fred/series/vintagedates?series_id=GDP&api_key=YOUR_KEY
```

**Use Case:** Track data revisions (important for backtesting).

---

## 4. Sources (Data Providers)

### `fred/sources`
Get all sources.

**Example:**
```
GET /fred/sources?api_key=YOUR_KEY
```

**Key Sources:**
- `1` - Board of Governors of the Federal Reserve System
- `4` - Bureau of Labor Statistics
- `18` - U.S. Energy Information Administration
- `19` - U.S. Department of Agriculture

### `fred/source`
Get a specific source.

**Example:**
```
GET /fred/source?source_id=1&api_key=YOUR_KEY
```

### `fred/source/releases`
Get all releases from a source.

**Example:**
```
GET /fred/source/releases?source_id=1&api_key=YOUR_KEY
```

**Use Case:** Get ALL Federal Reserve releases.

---

## 5. Tags (Discovery & Classification)

### `fred/tags`
Get all tags or search for tags.

**Example:**
```
GET /fred/tags?tag_names=monetary+policy&api_key=YOUR_KEY
```

**Key Tags:**
- `monetary policy`
- `interest rate`
- `federal funds`
- `exchange rate`
- `volatility`
- `agriculture`
- `energy`

### `fred/related_tags`
Get related tags for one or more tags.

**Example:**
```
GET /fred/related_tags?tag_names=monetary+policy&api_key=YOUR_KEY
```

**Use Case:** Discover related concepts (e.g., "monetary policy" → "interest rate", "federal funds").

### `fred/tags/series`
Get series matching tags.

**Example:**
```
GET /fred/tags/series?tag_names=monetary+policy;interest+rate&api_key=YOUR_KEY
```

**Use Case:** Find all series tagged with both "monetary policy" AND "interest rate".

---

## Implementation Strategy

### Phase 1: Backfill Historical Data (One-Time)

**Priority 1: Core Fed Policy (Bucket 4)**
```python
# H.15 Selected Interest Rates (Release 62)
releases = [62, 469, 456]  # H.15, STLFSI, NFCI
for release_id in releases:
    fetch_release_observations(release_id)
```

**Priority 2: FX Rates (Bucket 3)**
```python
# H.10 Foreign Exchange Rates (Release 17)
fetch_release_observations(17)
```

**Priority 3: Energy (Bucket 7)**
```python
# Petroleum Prices (Release 9)
fetch_release_observations(9)
```

### Phase 2: Daily Updates

**Use `fred/series/updates` to find new data:**
```python
# Get series updated today
updated_series = fetch_series_updates()

# Fetch only new observations
for series_id in updated_series:
    fetch_series_observations(
        series_id,
        observation_start=yesterday,
        observation_end=today
    )
```

### Phase 3: Discovery (Ongoing)

**Use tags to find new relevant series:**
```python
tags = ["soybean", "agriculture", "biodiesel", "renewable"]
for tag in tags:
    series_ids = fetch_tags_series(tag)
    # Add to collection list
```

---

## Key Series IDs (Quick Reference)

### Fed Policy (Bucket 4)
- `DFEDTARU` - Fed Funds Target Upper Limit
- `DFEDTARL` - Fed Funds Target Lower Limit
- `DFF` - Fed Funds Effective Rate
- `SOFR` - Secured Overnight Financing Rate
- `IORB` - Interest on Reserve Balances
- `DGS10` - 10-Year Treasury Constant Maturity
- `DGS2` - 2-Year Treasury Constant Maturity
- `T10Y2Y` - 10-Year minus 2-Year Treasury Spread

### Financial Stress (Bucket 8)
- `STLFSI` - St. Louis Fed Financial Stress Index
- `STLFSI4` - STLFSI 4-Week Moving Average
- `NFCI` - Chicago Fed National Financial Conditions Index
- `NFCILEVERAGE` - NFCI Leverage Subindex
- `NFCICREDIT` - NFCI Credit Subindex
- `VIXCLS` - CBOE Volatility Index (VIX)

### FX Rates (Bucket 3)
- `DEXBZUS` - Brazil / U.S. Foreign Exchange Rate
- `DEXCHUS` - China / U.S. Foreign Exchange Rate
- `DEXMXUS` - Mexico / U.S. Foreign Exchange Rate
- `DTWEXBGS` - Trade Weighted U.S. Dollar Index (Broad)

### Energy (Bucket 7)
- `DCOILWTICO` - Crude Oil Prices: West Texas Intermediate
- `GASREGW` - U.S. Regular All Formulations Gas Price
- `DHOILNYH` - No. 2 Heating Oil, New York Harbor

### Macro Context
- `UNRATE` - Unemployment Rate
- `CPIAUCSL` - Consumer Price Index for All Urban Consumers
- `GDP` - Gross Domestic Product
- `PAYEMS` - All Employees, Total Nonfarm

---

## Rate Limits & Best Practices

**Rate Limits:**
- 120 requests per minute
- No daily limit

**Best Practices:**
1. **Batch requests** - Use `/release/observations` to get multiple series at once
2. **Cache metadata** - Series info rarely changes
3. **Use `last_updated`** - Only fetch series that have new data
4. **Handle revisions** - Use `/series/vintagedates` for backtest accuracy
5. **Respect rate limits** - Add delays between requests if needed

---

## Error Handling

**Common Errors:**
- `400` - Bad request (invalid parameters)
- `404` - Series/release not found
- `429` - Rate limit exceeded
- `500` - FRED server error

**Retry Strategy:**
```python
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
```

---

## Data Quality Notes

**Missing Values:**
- Represented as `"."` in JSON/XML
- Skip or interpolate as needed

**Revisions:**
- GDP, employment data frequently revised
- Use `/series/vintagedates` to track revisions
- For backtesting, use point-in-time data

**Seasonal Adjustment:**
- Check `seasonal_adjustment` field
- Use NSA (Not Seasonally Adjusted) for raw data
- Use SA (Seasonally Adjusted) for trend analysis

---

## Integration with CBI-V15

**Current Implementation:**
- `trigger/FRED/Scripts/collect_fred_releases_historical.py` - Backfill script
- `config/fred_releases.yaml` - Release ID mapping to Big 8 buckets
- `raw.fred_economic` - Storage table in MotherDuck

**Daily Automation:**
```bash
# Run via daily script
bash scripts/daily_scrapecreators_run.sh

# Or standalone
python trigger/FRED/Scripts/collect_fred_releases_historical.py
```

**Expected Data Volume:**
- Initial backfill: 5-10 million observations
- Daily updates: 1,000-5,000 new observations
- Storage: ~500MB-1GB in DuckDB

---

## Additional Resources

- **FRED Homepage:** https://fred.stlouisfed.org/
- **API Documentation:** https://fred.stlouisfed.org/docs/api/fred/
- **FRASER (Historical Documents):** https://fraser.stlouisfed.org/
- **ALFRED (Vintage Data):** https://alfred.stlouisfed.org/
- **CASSIDI (Banking Data):** https://cassidi.stlouisfed.org/

---

**Last Updated:** December 16, 2025  
**Maintained By:** CBI-V15 Data Engineering Team


