-- Raw Tables: Immutable source data from ingestion

-- ScrapeCreators News (from collect.py)
CREATE TABLE IF NOT EXISTS raw.scrapecreators_news_buckets (
    article_id    TEXT PRIMARY KEY,
    bucket_name   TEXT NOT NULL,
    headline      TEXT,
    content       TEXT,
    source        TEXT,
    published_at  TIMESTAMP,
    created_at    TIMESTAMP DEFAULT current_timestamp
);

-- Databento Market Data (from ingest.py)
CREATE TABLE IF NOT EXISTS raw.databento_ohlcv_daily (
    as_of_date    DATE NOT NULL,
    symbol        TEXT NOT NULL,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    close         DOUBLE,
    volume        BIGINT,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (as_of_date, symbol)
);

-- Index for date-based queries
CREATE INDEX IF NOT EXISTS idx_databento_ohlcv_date ON raw.databento_ohlcv_daily (as_of_date);
