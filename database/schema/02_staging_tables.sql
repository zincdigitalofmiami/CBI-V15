-- Staging Tables: Cleaned, aligned, calendarized data

-- Aligned OHLCV with trading calendar
CREATE TABLE IF NOT EXISTS staging.ohlcv_daily (
    as_of_date    DATE NOT NULL,
    symbol        TEXT NOT NULL,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    close         DOUBLE,
    volume        BIGINT,
    is_trading_day BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (as_of_date, symbol)
);

-- Cleaned news with sentiment
CREATE TABLE IF NOT EXISTS staging.news_daily (
    as_of_date    DATE NOT NULL,
    article_id    TEXT NOT NULL,
    bucket_name   TEXT NOT NULL,
    headline      TEXT,
    sentiment     DOUBLE,
    PRIMARY KEY (as_of_date, article_id)
);

-- Crush data (from external sources)
CREATE TABLE IF NOT EXISTS staging.crush_daily (
    as_of_date              DATE PRIMARY KEY,
    nopa_volume_us          DOUBLE,
    capacity_utilization    DOUBLE
);

-- China demand data
CREATE TABLE IF NOT EXISTS staging.china_daily (
    as_of_date              DATE PRIMARY KEY,
    soybean_imports_mmt     DOUBLE
);
