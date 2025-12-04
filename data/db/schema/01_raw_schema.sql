-- MotherDuck Schema: raw
-- Vendor data as-delivered, minimal casting, no business logic

-- Databento futures OHLCV
CREATE TABLE IF NOT EXISTS raw.databento_futures_ohlcv_1d (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    open_interest BIGINT,
    source VARCHAR DEFAULT 'databento',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);

-- FRED economic indicators
CREATE TABLE IF NOT EXISTS raw.fred_economic (
    series_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    value DOUBLE,
    source VARCHAR DEFAULT 'fred',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (series_id, date)
);

-- ScrapeCreators news buckets
CREATE TABLE IF NOT EXISTS raw.scrapecreators_news_buckets (
    article_id VARCHAR PRIMARY KEY,
    published_date DATE NOT NULL,
    bucket VARCHAR NOT NULL,
    headline TEXT,
    content TEXT,
    sentiment_score DECIMAL(5, 4),
    source VARCHAR DEFAULT 'scrapecreators',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ScrapeCreators Trump/policy feed
CREATE TABLE IF NOT EXISTS raw.scrapecreators_trump (
    post_id VARCHAR PRIMARY KEY,
    published_date TIMESTAMP NOT NULL,
    platform VARCHAR,
    content TEXT,
    sentiment_score DECIMAL(5, 4),
    zl_impact_score DECIMAL(5, 4),
    source VARCHAR DEFAULT 'scrapecreators',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NOAA weather data
CREATE TABLE IF NOT EXISTS raw.weather_noaa (
    station_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    tavg_c DECIMAL(5, 2),
    tmin_c DECIMAL(5, 2),
    tmax_c DECIMAL(5, 2),
    prcp_mm DECIMAL(6, 2),
    snow_mm DECIMAL(6, 2),
    region VARCHAR,
    country VARCHAR,
    source VARCHAR DEFAULT 'noaa',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (station_id, date)
);

-- CFTC Commitments of Traders
CREATE TABLE IF NOT EXISTS raw.cftc_cot (
    report_date DATE NOT NULL,
    commodity VARCHAR NOT NULL,
    noncommercial_long BIGINT,
    noncommercial_short BIGINT,
    commercial_long BIGINT,
    commercial_short BIGINT,
    total_open_interest BIGINT,
    source VARCHAR DEFAULT 'cftc',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, commodity)
);

-- USDA WASDE reports
CREATE TABLE IF NOT EXISTS raw.usda_wasde (
    report_date DATE NOT NULL,
    commodity VARCHAR NOT NULL,
    country VARCHAR,
    metric VARCHAR NOT NULL,
    value DOUBLE,
    unit VARCHAR,
    source VARCHAR DEFAULT 'usda_wasde',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, commodity, country, metric)
);

-- USDA Export Sales
CREATE TABLE IF NOT EXISTS raw.usda_export_sales (
    report_date DATE NOT NULL,
    commodity VARCHAR NOT NULL,
    destination_country VARCHAR,
    net_sales_mt DOUBLE,
    exports_mt DOUBLE,
    outstanding_sales_mt DOUBLE,
    source VARCHAR DEFAULT 'usda_exports',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, commodity, destination_country)
);

-- EIA Biofuels data
CREATE TABLE IF NOT EXISTS raw.eia_biofuels (
    date DATE NOT NULL,
    series_id VARCHAR NOT NULL,
    value DOUBLE,
    unit VARCHAR,
    region VARCHAR,
    source VARCHAR DEFAULT 'eia',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, series_id)
);

