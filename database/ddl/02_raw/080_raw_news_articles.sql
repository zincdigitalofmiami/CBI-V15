-- Raw News Articles
-- ScrapeCreators news buckets + Trump/policy feed

CREATE TABLE IF NOT EXISTS raw.scrapecreators_news_buckets (
    article_id VARCHAR PRIMARY KEY,
    date DATE NOT NULL,  -- Publication date (macros use this)
    published_at TIMESTAMP, -- Full timestamp from source
    
    -- Content
    headline TEXT,
    content TEXT,
    url TEXT,
    author VARCHAR,
    
    -- Classification
    bucket_name VARCHAR NOT NULL,  -- 'biofuel_policy', 'china_demand', etc.
    edition_type VARCHAR,          -- 'pre_open', 'post_close', etc.
    source VARCHAR DEFAULT 'scrapecreators',
    source_trust_score DECIMAL(3, 2),
    
    -- Downstream Enrichments (Populated by Agents)
    sentiment_score DECIMAL(5, 4),
    zl_sentiment VARCHAR,  -- 'BULLISH_ZL', 'BEARISH_ZL', 'NEUTRAL'
    is_trump_related BOOLEAN DEFAULT FALSE,
    policy_axis VARCHAR,  -- 'TRADE_CHINA', 'TRADE_TARIFFS', etc.
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    ON raw.scrapecreators_news_buckets(bucket_name);
CREATE INDEX IF NOT EXISTS idx_news_date 
    ON raw.scrapecreators_news_buckets(date);
