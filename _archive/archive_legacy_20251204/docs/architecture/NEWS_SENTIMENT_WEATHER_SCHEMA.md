# News / Sentiment / Weather Schema Specification

**Date**: November 29, 2025  
**Status**: ✅ **CANONICAL** - Single source of truth  
**Architecture**: Python-first (post-Dataform)

---

## 1. News / Sentiment Pipeline

### 1.1 Raw Layer: `raw.scrapecreators_news_buckets`

**Schema** (Aligned with ScrapeCreators API):
```sql
CREATE TABLE `cbi-v15.raw.scrapecreators_news_buckets` (
  date DATE,
  article_id STRING,
  theme_primary STRING,           -- Renamed from bucket_type (source has theme)
  is_trump_related BOOL,
  policy_axis STRING,              -- TRADE_CHINA, ENERGY_POLICY, etc.
  horizon STRING,                  -- SHORT_TERM, MEDIUM_TERM, LONG_TERM
  zl_sentiment STRING,             -- BULLISH_ZL, BEARISH_ZL, NEUTRAL
  impact_magnitude STRING,         -- HIGH, MEDIUM, LOW (enum validated)
  sentiment_confidence FLOAT64,    -- 0.0-1.0
  sentiment_raw_score FLOAT64,     -- -1.0 to 1.0
  headline STRING,                 -- Keep for audit, not queried by default
  content STRING,                  -- LARGE TEXT - avoid select *
  source STRING,
  source_trust_score FLOAT64,
  created_at TIMESTAMP
)
PARTITION BY DATE(date)
CLUSTER BY theme_primary, is_trump_related;
```

**Key Decisions**:
- `theme_primary` is the source field (not `bucket_type`)
- `headline` and `content` are TEXT columns - **avoid select \*** to control costs
- Partitioned by DATE, clustered by theme/trump for pruning

**Validation Rules**:
```python
REQUIRED_FIELDS = ['date', 'article_id', 'theme_primary', 'zl_sentiment', 'impact_magnitude']
VALID_SENTIMENTS = ['BULLISH_ZL', 'BEARISH_ZL', 'NEUTRAL']
VALID_IMPACTS = ['HIGH', 'MEDIUM', 'LOW']
```

---

### 1.2 Staging Layer: `staging.news_bucketed`

**Schema** (Aggregated by date + theme):
```sql
CREATE TABLE `cbi-v15.staging.news_bucketed` (
  date DATE,
  theme STRING,                    -- Aligned with theme_primary from raw
  article_count INT64,
  avg_sentiment_score FLOAT64,     -- Mean of sentiment_raw_score
  avg_confidence FLOAT64,          -- Mean of sentiment_confidence
  high_impact_count INT64,
  medium_impact_count INT64,
  low_impact_count INT64,
  bullish_count INT64,
  bearish_count INT64,
  neutral_count INT64
)
PARTITION BY DATE(date)
CLUSTER BY theme;
```

**Aggregation Logic** (Python):
```python
GROUP BY date, theme_primary AS theme
AVG(sentiment_raw_score) AS avg_sentiment_score
AVG(sentiment_confidence) AS avg_confidence
COUNT(*) AS article_count
COUNTIF(impact_magnitude='HIGH') AS high_impact_count
COUNTIF(zl_sentiment='BULLISH_ZL') AS bullish_count
```

---

### 1.3 Staging Layer: `staging.sentiment_buckets`

**Schema** (ZL sentiment aggregates):
```sql
CREATE TABLE `cbi-v15.staging.sentiment_buckets` (
  date DATE,
  theme STRING,
  bullish_count INT64,
  bearish_count INT64,
  neutral_count INT64,
  net_sentiment INT64,            -- bullish - bearish
  sentiment_ratio FLOAT64,        -- bullish / (bullish + bearish)
  weighted_score FLOAT64          -- avg(sentiment_raw_score * sentiment_confidence)
)
PARTITION BY DATE(date)
CLUSTER BY theme;
```

**Aggregation Logic**:
```python
GROUP BY date, theme_primary AS theme
COUNT(IF zl_sentiment='BULLISH_ZL') AS bullish_count
COUNT(IF zl_sentiment='BEARISH_ZL') AS bearish_count  
COUNT(IF zl_sentiment='NEUTRAL') AS neutral_count
bullish_count - bearish_count AS net_sentiment
bullish_count / (bullish_count + bearish_count) AS sentiment_ratio
AVG(sentiment_raw_score * sentiment_confidence) AS weighted_score
```

---

## 2. Weather Pipeline

### 2.1 Raw Layer: `raw.weather_noaa`

**Schema** (Generic metric/value pairs):
```sql
CREATE TABLE `cbi-v15.raw.weather_noaa` (
  date DATE,
  station_id STRING,
  region STRING,                   -- US_MIDWEST, BRAZIL_SOUTH, ARGENTINA_PAMPAS
  metric STRING,                   -- TEMP_MAX_C, PRECIP_MM, SOIL_MOISTURE_PCT
  value FLOAT64,
  unit STRING,                     -- Added for clarity
  source STRING                    -- NOAA, INMET, SMN
)
PARTITION BY DATE(date)
CLUSTER BY region, metric;
```

**Validation Rules**:
```python
REQUIRED_FIELDS = ['date', 'station_id', 'region', 'metric', 'value']
VALID_REGIONS = ['US_MIDWEST', 'US_SOUTH', 'BRAZIL_SOUTH', 'BRAZIL_CENTER_WEST', 
                 'ARGENTINA_PAMPAS', 'ARGENTINA_CHACO']
VALID_METRICS = ['TEMP_MAX_C', 'TEMP_MIN_C', 'TEMP_MEAN_C', 'PRECIP_MM', 
                 'SOIL_MOISTURE_PCT', 'GDD_BASE_10C']
```

---

### 2.2 Staging Layer: `staging.weather_regions_aggregated`

**Schema** (Daily regional aggregates):
```sql
CREATE TABLE `cbi-v15.staging.weather_regions_aggregated` (
  date DATE,
  region STRING,
  metric STRING,
  value_mean FLOAT64,              -- Regional mean (primary)
  value_median FLOAT64,            -- Regional median
  value_min FLOAT64,
  value_max FLOAT64,
  station_count INT64,             -- Number of stations reporting
  coverage_pct FLOAT64             -- % of expected stations
)
PARTITION BY DATE(date)
CLUSTER BY region, metric;
```

**Aggregation Logic**:
```python
GROUP BY date, region, metric
AVG(value) AS value_mean
APPROX_QUANTILES(value, 2)[OFFSET(1)] AS value_median
MIN(value) AS value_min
MAX(value) AS value_max
COUNT(DISTINCT station_id) AS station_count
station_count / expected_station_count * 100 AS coverage_pct
```

**Station Weighting** (Optional):
- Equal weight for now (simple mean)
- Future: weight by station reliability/coverage history

---

## 3. Pipeline Flow

### News/Sentiment
```
ScrapeCreators API
    ↓ [Python ingestion]
raw.scrapecreators_news_buckets (TEXT columns, avoid select *)
    ↓ [Python: build_news_staging.py]
staging.news_bucketed (aggregated by date + theme)
staging.sentiment_buckets (ZL sentiment metrics)
    ↓ [Feature builder - future]
features.sentiment_daily (rolling windows, regime-adjusted)
```

### Weather
```
NOAA/INMET/SMN APIs
    ↓ [Python ingestion]
raw.weather_noaa (station-level, metric/value pairs)
    ↓ [Python: build_weather_staging.py]
staging.weather_regions_aggregated (daily regional means)
    ↓ [Feature builder - future]
features.weather_daily (GDD, precip anomalies, regime-adjusted)
```

---

## 4. Cost Control

### News (Large TEXT columns)
**Problem**: `headline` (avg 100 bytes) and `content` (avg 2KB) add up.

**Solutions**:
1. **Never select * from raw.scrapecreators_news_buckets**
2. Query pattern:
   ```sql
   SELECT date, article_id, theme_primary, zl_sentiment, impact_magnitude
   FROM raw.scrapecreators_news_buckets  -- No headline/content
   ```
3. If full text needed: Store in GCS, keep only article_id + GCS URI in BQ

### Weather (Generic metric/value)
**Problem**: Schema doesn't enforce consistent units/metrics.

**Solutions**:
1. Validate `metric` against VALID_METRICS enum at ingestion
2. Validate `unit` matches metric (e.g., TEMP_MAX_C → Celsius)
3. Fail fast on unknown metrics

---

## 5. Governance

### Schema Assertions
Add to `scripts/validation/check_schemas.py`:
```python
def validate_news_raw(df):
    assert set(REQUIRED_FIELDS).issubset(df.columns)
    assert df['zl_sentiment'].isin(VALID_SENTIMENTS).all()
    assert df['impact_magnitude'].isin(VALID_IMPACTS).all()
    
def validate_weather_raw(df):
    assert set(REQUIRED_FIELDS).issubset(df.columns)
    assert df['region'].isin(VALID_REGIONS).all()
    assert df['metric'].isin(VALID_METRICS).all()
```

### Single Source of Truth
- **This document** is the canonical schema reference
- Update skeleton SQL to match this spec
- Remove conflicting definitions from archived Dataform files

---

## 6. Implementation Checklist

- [ ] Update `scripts/setup/create_skeleton_tables.sql` with aligned schemas
- [ ] Create `src/staging/build_news_staging.py` (news_bucketed + sentiment_buckets)
- [ ] Create `src/staging/build_weather_staging.py` (weather_regions_aggregated)
- [ ] Create `scripts/validation/check_schemas.py` (assertions)
- [ ] Add to main feature builder (future: sentiment/weather features)
- [ ] Document in `docs/ingestion/NEWS_SENTIMENT_WEATHER.md`

---

**Last Updated**: November 29, 2025





