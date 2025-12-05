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
