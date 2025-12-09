-- News Signals (OpenAI Agent Output)
-- Processed news articles with trading signals extracted by AI

CREATE TABLE IF NOT EXISTS features.news_signals (
    -- Primary Key
    article_id TEXT PRIMARY KEY,
    
    -- Sentiment Analysis
    zl_sentiment TEXT,  -- BULLISH_ZL, BEARISH_ZL, NEUTRAL
    sentiment_confidence FLOAT,
    sentiment_reasoning TEXT,
    
    -- Impact Assessment
    impact_magnitude TEXT,  -- HIGH, MEDIUM, LOW
    impact_reasoning TEXT,
    
    -- Time Horizon
    horizon TEXT,  -- FLASH, TACTICAL, STRUCTURAL
    horizon_reasoning TEXT,
    
    -- Thematic Classification
    theme_primary TEXT,  -- SUPPLY_WEATHER, DEMAND_BIOFUELS, DEMAND_CHINA, TRADE_GEO, etc.
    
    -- Policy Axis
    policy_axis TEXT,  -- TRADE_CHINA, TRADE_TARIFFS, BIOFUELS_RFS, etc.
    
    -- Entities (JSON)
    entities_commodities TEXT[],
    entities_countries TEXT[],
    entities_companies TEXT[],
    entities_policies TEXT[],
    
    -- Price Targets (JSON)
    price_targets JSON,
    
    -- Processing Metadata
    processed_at TIMESTAMP DEFAULT current_timestamp,
    model_version TEXT DEFAULT 'gpt-4o'
);

-- Index for sentiment queries
CREATE INDEX IF NOT EXISTS idx_news_signals_sentiment ON features.news_signals(zl_sentiment, impact_magnitude);

-- Index for theme queries
CREATE INDEX IF NOT EXISTS idx_news_signals_theme ON features.news_signals(theme_primary, horizon);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_news_signals_processed ON features.news_signals(processed_at);

