-- Raw TradingEconomics Data
-- Economic calendar events and indicators

CREATE TABLE IF NOT EXISTS raw.tradingeconomics_calendar (
    event_id VARCHAR PRIMARY KEY,
    event_date TIMESTAMP NOT NULL,
    country VARCHAR,
    category VARCHAR,  -- 'GDP', 'Interest Rate', 'Trade Balance', etc.
    event_name VARCHAR NOT NULL,
    actual DOUBLE,
    forecast DOUBLE,
    previous DOUBLE,
    unit VARCHAR,
    importance VARCHAR,  -- 'high', 'medium', 'low'
    source VARCHAR DEFAULT 'tradingeconomics',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw.tradingeconomics_indicators (
    indicator_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    country VARCHAR,
    indicator_name VARCHAR NOT NULL,
    value DOUBLE,
    unit VARCHAR,
    source VARCHAR DEFAULT 'tradingeconomics',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicator_id, date)
);

-- Key events: USDA reports, FOMC, China trade data, Brazil GDP

CREATE INDEX IF NOT EXISTS idx_te_calendar_date 
    ON raw.tradingeconomics_calendar(event_date);
CREATE INDEX IF NOT EXISTS idx_te_calendar_category 
    ON raw.tradingeconomics_calendar(category);

