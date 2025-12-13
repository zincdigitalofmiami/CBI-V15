-- Staging: Sentiment Buckets
-- Daily sentiment scores by Big 8 bucket

CREATE TABLE IF NOT EXISTS staging.sentiment_buckets (
    date DATE PRIMARY KEY,
    -- Big 8 bucket scores (0-100)
    crush_sentiment DECIMAL(5, 2),
    china_sentiment DECIMAL(5, 2),
    fx_sentiment DECIMAL(5, 2),
    fed_sentiment DECIMAL(5, 2),
    tariff_sentiment DECIMAL(5, 2),
    biofuel_sentiment DECIMAL(5, 2),
    energy_sentiment DECIMAL(5, 2),
    volatility_sentiment DECIMAL(5, 2),
    -- Aggregate
    composite_sentiment DECIMAL(5, 2),
    -- Confidence/coverage
    coverage_pct DECIMAL(5, 4),  -- % of buckets with data
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment scores derived from news_bucketed + policy signals

