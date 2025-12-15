-- Raw FRED Economic Indicators
-- Federal Reserve Economic Data (60+ series)

CREATE TABLE IF NOT EXISTS raw.fred_economic (
    series_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    value DOUBLE,
    source VARCHAR DEFAULT 'fred',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (series_id, date)
);

-- Raw FRED Series Metadata
-- Created by seed harvest script
CREATE TABLE IF NOT EXISTS raw.fred_series_metadata (
    series_id VARCHAR PRIMARY KEY,
    title VARCHAR,
    frequency VARCHAR,
    units VARCHAR,
    seasonal_adjustment VARCHAR,
    last_updated TIMESTAMP,
    notes TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Key series for ZL forecasting:
-- Rates: FEDFUNDS, DGS10, DGS2, T10Y2Y
-- FX: DEXUSEU, DEXBZUS, DEXCHUS, DEXMXUS, DTWEXBGS
-- Volatility: VIXCLS, STLFSI4
-- Macro: CPIAUCSL, UNRATE, NFCI
-- Commodities: DCOILWTICO (WTI crude)

CREATE INDEX IF NOT EXISTS idx_fred_economic_series 
    ON raw.fred_economic(series_id);
