# Intelligent News Pipeline - AI-Powered News Processing

**Purpose:** End-to-end news ingestion and signal extraction using Anchor browser automation and OpenAI Agents.

**Architecture:**

1. **Anchor Browser Scraping** - Authenticated scraping with JavaScript rendering
2. **OpenAI Agent Processing** - AI-powered sentiment analysis and signal extraction
3. **MotherDuck Storage** - Structured storage of signals for feature engineering

---

## Components

### 1. Anchor Browser Scraper

**Purpose:** Scrape ProFarmer with authenticated browser automation

**Features:**

- Handles login flows automatically
- Renders JavaScript-heavy pages
- AI-powered element selection
- Extracts full article content (500 words max)

**Reference:** Anchor browser automation patterns (project-local scripts)

---

### 2. OpenAI Agent Signal Processor

**Purpose:** Extract trading signals from news articles using GPT-4o

**Signal Schema:**

```typescript
{
  article_id: string,

  // Sentiment
  zl_sentiment: "BULLISH_ZL" | "BEARISH_ZL" | "NEUTRAL",
  sentiment_confidence: number,  // 0-1
  sentiment_reasoning: string,

  // Impact
  impact_magnitude: "HIGH" | "MEDIUM" | "LOW",
  impact_reasoning: string,

  // Time Horizon
  horizon: "FLASH" | "TACTICAL" | "STRUCTURAL",
  horizon_reasoning: string,

  // Theme
  theme_primary: "SUPPLY_WEATHER" | "DEMAND_BIOFUELS" | "DEMAND_CHINA" | ...,

  // Entities
  entities: {
    commodities: string[],
    countries: string[],
    companies: string[],
    policies: string[]
  },

  // Price Targets (if mentioned)
  price_targets: Array<{
    commodity: string,
    target: number,
    timeframe: string
  }>
}
```

**AI Prompt Context:**

- ZL (soybean oil) fundamentals
- BULLISH drivers: China demand ↑, biofuel mandates ↑, crude oil ↑, USD ↓
- BEARISH drivers: China demand ↓, biofuel mandates ↓, crude oil ↓, USD ↑
- Related commodities: ZS, ZM, CL, HG

---

### 3. Pipeline Orchestrator

**Phases:**

1. **Scrape** - Anchor browser automation
2. **Process** - OpenAI Agent signal extraction
3. **Store** - MotherDuck features table
4. **Run** - Downstream feature engineering

**Schedules:**

- **Every 6 hours** - Continuous news monitoring
- **Pre-market (6 AM UTC)** - Overnight news
- **Post-market (9 PM UTC)** - Closing news

---

## Database Schema

### `features.news_signals`

```sql
CREATE TABLE features.news_signals (
    article_id TEXT PRIMARY KEY,

    -- Sentiment
    zl_sentiment TEXT,
    sentiment_confidence FLOAT,
    sentiment_reasoning TEXT,

    -- Impact
    impact_magnitude TEXT,
    impact_reasoning TEXT,

    -- Horizon
    horizon TEXT,
    horizon_reasoning TEXT,

    -- Theme
    theme_primary TEXT,
    policy_axis TEXT,

    -- Entities
    entities_commodities TEXT[],
    entities_countries TEXT[],
    entities_companies TEXT[],
    entities_policies TEXT[],

    -- Price Targets
    price_targets JSON,

    -- Metadata
    processed_at TIMESTAMP,
    model_version TEXT DEFAULT 'gpt-4o'
);
```

---

## Jobs

| Job                   | File                                      | Status     |
| --------------------- | ----------------------------------------- | ---------- |
| Anchor Scraper        | `src/ingestion/usda/profarmer_anchor.py` | ✅ Created |
| Signal Processing     | `src/ingestion/scrapecreators/sentiment_calculator.py` | ✅ Created |
| Pipeline Runner       | `src/ingestion/scrapecreators/collect_all_scrapecreators.py` | ✅ Created |

---

## Monitoring

### MotherDuck Queries

```sql
-- Check signal distribution
SELECT zl_sentiment, impact_magnitude, COUNT(*) as count
FROM features.news_signals
GROUP BY zl_sentiment, impact_magnitude;

-- Recent high-impact signals
SELECT article_id, zl_sentiment, sentiment_confidence, impact_magnitude
FROM features.news_signals
WHERE impact_magnitude = 'HIGH'
  AND processed_at >= NOW() - INTERVAL '24 hours'
ORDER BY processed_at DESC;
```

---

**Last Updated:** December 10, 2025
