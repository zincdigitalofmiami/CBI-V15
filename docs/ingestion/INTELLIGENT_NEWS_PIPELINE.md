# Intelligent News Pipeline - AI-Powered News Processing

**Purpose:** End-to-end news ingestion and signal extraction using Anchor browser automation and OpenAI Agents.

**Architecture:**
1. **Anchor Browser Scraping** - Authenticated scraping with JavaScript rendering
2. **OpenAI Agent Processing** - AI-powered sentiment analysis and signal extraction
3. **MotherDuck Storage** - Structured storage of signals for feature engineering

---

## Components

### 1. Anchor Browser Scraper (`trigger/profarmer_anchor_scraper.ts`)

**Purpose:** Scrape ProFarmer with authenticated browser automation

**Features:**
- Handles login flows automatically
- Renders JavaScript-heavy pages
- AI-powered element selection
- Extracts full article content (500 words max)

**Based on:** [Trigger.dev Anchor Browser Example](https://trigger.dev/docs/guides/example-projects/anchor-browser-web-scraper)

**Sections Scraped:**
- First Thing Today (pre_open)
- Ahead of the Open (pre_open)
- After the Bell (post_close)
- Agriculture News (intraday)
- Newsletters (newsletter)

**Trigger:**
```bash
npx trigger.dev@latest trigger profarmer-anchor-scraper
```

---

### 2. OpenAI Agent Signal Processor (`trigger/news_to_signals_openai_agent.ts`)

**Purpose:** Extract trading signals from news articles using GPT-4o

**Based on:**
- [OpenAI Agent SDK with Guardrails](https://trigger.dev/docs/guides/example-projects/openai-agent-sdk-guardrails)
- [Vercel AI SDK](https://trigger.dev/docs/guides/examples/vercel-ai-sdk)

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

**Guardrails:**
- Temperature: 0.3 (consistent analysis)
- Structured output with Zod schema validation
- Reasoning required for all classifications

**Trigger:**
```bash
npx trigger.dev@latest trigger news-to-signals-openai-agent --payload '{"batchSize": 50}'
```

---

### 3. Intelligent News Pipeline Orchestrator (`trigger/intelligent_news_pipeline.ts`)

**Purpose:** End-to-end orchestration of scraping → processing → storage

**Phases:**
1. **Scrape** - Anchor browser automation
2. **Process** - OpenAI Agent signal extraction
3. **Store** - MotherDuck features table
4. **Trigger** - Downstream feature engineering (TODO)

**Schedules:**
- **Every 6 hours** - Continuous news monitoring
- **Pre-market (6 AM UTC)** - Overnight news
- **Post-market (9 PM UTC)** - Closing news

**Trigger:**
```bash
npx trigger.dev@latest trigger intelligent-news-pipeline
```

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

## Next.js Webhook Integration

**File:** `dashboard/app/api/trigger/route.ts`

**Purpose:** Receive job status updates from Trigger.dev

**Events Handled:**
- `job.completed` - Log results, trigger downstream jobs
- `job.failed` - Send alerts, retry if appropriate
- `job.started` - Update dashboard status

**Webhook URL:** `https://your-dashboard.vercel.app/api/trigger`

**Configuration in Trigger.dev:**
1. Go to Project Settings → Webhooks
2. Add webhook URL
3. Select events: `job.completed`, `job.failed`, `job.started`
4. Save

---

## Environment Variables

**Required:**
```bash
# Trigger.dev
TRIGGER_SECRET_KEY="tr_dev_5cabtqdvsHwK8L9sQqRi"

# OpenAI
OPENAI_API_KEY=your_openai_key

# Anchor Browser
ANCHOR_API_KEY=your_anchor_key

# ProFarmer
PROFARMER_USERNAME=your_username
PROFARMER_PASSWORD=your_password

# MotherDuck
MOTHERDUCK_TOKEN=your_token
MOTHERDUCK_DB=cbi_v15
```

---

## Setup Instructions

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment
```bash
cp .env.local.example .env.local
# Add all API keys
```

### 3. Initialize Trigger.dev
```bash
npx trigger.dev@latest init
```

### 4. Start Dev Server
```bash
npx trigger.dev@latest dev
```

### 5. Test Pipeline
```bash
# Test Anchor scraper
npx trigger.dev@latest trigger profarmer-anchor-scraper

# Test OpenAI agent
npx trigger.dev@latest trigger news-to-signals-openai-agent

# Test full pipeline
npx trigger.dev@latest trigger intelligent-news-pipeline
```

### 6. Deploy
```bash
npx trigger.dev@latest deploy
```

---

## Monitoring

### Trigger.dev Dashboard
- View runs: https://cloud.trigger.dev/projects/cbi-v15/runs
- Check logs and errors
- Retry failed jobs

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

**Last Updated:** December 9, 2024

