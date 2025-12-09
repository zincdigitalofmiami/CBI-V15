-- TradingEconomics Commodity Data
-- Daily snapshots of commodity prices, forecasts, and analysis

CREATE TABLE IF NOT EXISTS raw.tradingeconomics_commodities (
    -- Primary Key
    commodity TEXT,
    scraped_at TIMESTAMP,
    
    -- Metadata
    url TEXT,
    bucket_name TEXT,
    priority TEXT,
    
    -- Price Data
    current_price DOUBLE,
    price_change DOUBLE,
    price_change_pct DOUBLE,
    
    -- Historical Range
    high_52w DOUBLE,
    low_52w DOUBLE,
    
    -- Forecasts
    forecast_1m DOUBLE,
    forecast_3m DOUBLE,
    forecast_1y DOUBLE,
    
    -- News & Analysis
    latest_news TEXT,
    analysis TEXT,
    
    PRIMARY KEY (commodity, scraped_at)
);

-- Index for commodity queries
CREATE INDEX IF NOT EXISTS idx_te_commodity ON raw.tradingeconomics_commodities(commodity, scraped_at);

-- Index for bucket queries
CREATE INDEX IF NOT EXISTS idx_te_bucket ON raw.tradingeconomics_commodities(bucket_name, scraped_at);

-- Index for priority queries
CREATE INDEX IF NOT EXISTS idx_te_priority ON raw.tradingeconomics_commodities(priority, scraped_at);

