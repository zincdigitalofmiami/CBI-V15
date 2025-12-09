-- Bucket-Level News (Direct Scraping)
-- Separate from scrapecreators_news_buckets (API-based)
-- Sources: ProFarmer, Agrimoney, CONAB, Reuters, Farm Bureau, State Ag Depts, etc.

CREATE TABLE IF NOT EXISTS raw.bucket_news (
    -- Primary Keys
    date DATE,
    article_id TEXT PRIMARY KEY,
    
    -- Content
    headline TEXT,
    content TEXT,
    author TEXT,
    
    -- Source Metadata
    source TEXT,  -- "ProFarmer", "Agrimoney China", "Farm Bureau", etc.
    source_trust_score FLOAT,
    url TEXT,
    
    -- Bucket Assignment
    bucket_name TEXT,  -- "china", "tariff", "biofuel", "weather", "profarmer_anchor", etc.
    
    -- ProFarmer-Specific
    edition_type TEXT,  -- "pre_open", "post_close", "intraday", "newsletter" (NULL for non-ProFarmer)
    
    -- Timestamps
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- Index for deduplication
CREATE INDEX IF NOT EXISTS idx_bucket_news_url ON raw.bucket_news(url);

-- Index for bucket queries
CREATE INDEX IF NOT EXISTS idx_bucket_news_bucket ON raw.bucket_news(bucket_name, date);

