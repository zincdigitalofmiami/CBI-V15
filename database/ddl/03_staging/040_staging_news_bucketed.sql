-- Staging: News Bucketed
-- Aggregated daily news sentiment by bucket

CREATE TABLE IF NOT EXISTS staging.news_bucketed (
    date DATE NOT NULL,
    bucket VARCHAR NOT NULL,  -- Big 8 buckets
    article_count INT,
    avg_sentiment DECIMAL(5, 4),
    max_sentiment DECIMAL(5, 4),
    min_sentiment DECIMAL(5, 4),
    sentiment_std DECIMAL(5, 4),
    -- Derived signals
    sentiment_momentum_3d DECIMAL(5, 4),
    extreme_positive_count INT,  -- sentiment > 0.7
    extreme_negative_count INT,  -- sentiment < -0.7
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, bucket)
);

-- Buckets: crush, china, fx, fed, tariff, biofuel, energy, volatility

CREATE INDEX IF NOT EXISTS idx_news_bucketed_bucket 
    ON staging.news_bucketed(bucket);

