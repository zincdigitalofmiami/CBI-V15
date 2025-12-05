-- Cleaned news with sentiment
CREATE TABLE IF NOT EXISTS staging.news_daily (
    as_of_date    DATE NOT NULL,
    article_id    TEXT NOT NULL,
    bucket_name   TEXT NOT NULL,
    headline      TEXT,
    sentiment     DOUBLE,
    PRIMARY KEY (as_of_date, article_id)
);
