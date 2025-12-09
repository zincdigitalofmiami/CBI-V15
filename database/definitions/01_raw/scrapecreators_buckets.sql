-- ScrapeCreators News Buckets - Full 3-Way Segmentation Schema
-- Thematic (what it's about) + Time-Horizon (how long it matters) + Impact/Sentiment (ZL direction)
CREATE TABLE IF NOT EXISTS raw.scrapecreators_news_buckets (
    -- Primary Keys
    date DATE,
    article_id TEXT PRIMARY KEY,

    -- Thematic Bucket (PRIMARY)
    theme_primary TEXT,  -- SUPPLY_WEATHER, DEMAND_BIOFUELS, TRADE_GEO, MACRO_FX, LOGISTICS, POSITIONING, IDIOSYNCRATIC

    -- Trump-Specific Tags
    is_trump_related BOOLEAN,
    policy_axis TEXT,  -- TRADE_CHINA, TRADE_TARIFFS, BIOFUELS_RFS, EPA_REGS, AGRICULTURE_SUBSIDY, GEOPOLITICS_SOY_ROUTE, TRUMP_SOCIAL

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

    -- Tracking & Metadata
    url TEXT,  -- Source URL for deduplication
    search_query TEXT,  -- Which search query found this (for debugging/tuning)

    -- Timestamps
    created_at TIMESTAMP DEFAULT current_timestamp
);
