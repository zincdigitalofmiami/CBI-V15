-- Raw News Articles
-- ScrapeCreators news buckets + Trump/policy feed

CREATE TABLE IF NOT EXISTS raw.scrapecreators_news_buckets (
    article_id VARCHAR PRIMARY KEY,
    date DATE NOT NULL,  -- Publication date (macros use this)
    bucket VARCHAR NOT NULL,  -- 'biofuel_policy', 'china_demand', 'tariffs_trade_policy', etc.
    headline TEXT,
    content TEXT,
    sentiment_score DECIMAL(5, 4),
    zl_sentiment VARCHAR,  -- 'BULLISH_ZL', 'BEARISH_ZL', 'NEUTRAL' (macro uses this)
    is_trump_related BOOLEAN DEFAULT FALSE,  -- For tariff bucket filtering
    policy_axis VARCHAR,  -- 'TRADE_CHINA', 'TRADE_TARIFFS', etc.
    source_name VARCHAR,
    source VARCHAR DEFAULT 'scrapecreators',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw.scrapecreators_trump (
    post_id VARCHAR PRIMARY KEY,
    published_date TIMESTAMP NOT NULL,
    platform VARCHAR,  -- 'truth_social', 'twitter'
    content TEXT,
    sentiment_score DECIMAL(5, 4),
    zl_impact_score DECIMAL(5, 4),
    source VARCHAR DEFAULT 'scrapecreators',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_bucket 
    ON raw.scrapecreators_news_buckets(bucket);
CREATE INDEX IF NOT EXISTS idx_news_date 
    ON raw.scrapecreators_news_buckets(date);

