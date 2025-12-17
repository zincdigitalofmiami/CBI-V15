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

-- Key series for ZL forecasting:
-- Rates: FEDFUNDS, DGS10, DGS2, T10Y2Y
-- FX: DEXUSEU, DEXBZUS, DEXCHUS, DEXMXUS, DTWEXBGS
-- Volatility: VIXCLS, STLFSI4
-- Macro: CPIAUCSL, UNRATE, NFCI
-- Commodities: DCOILWTICO (WTI crude)

CREATE INDEX IF NOT EXISTS idx_fred_economic_series 
    ON raw.fred_economic(series_id);

-- Raw FRED Series Metadata
-- Discovered/seeded series catalog for expanding raw.fred_economic coverage

CREATE TABLE IF NOT EXISTS raw.fred_series_metadata (
    series_id VARCHAR PRIMARY KEY,
    title TEXT,
    category VARCHAR,
    frequency VARCHAR,
    units VARCHAR,
    seasonal_adjustment VARCHAR,
    observation_start DATE,
    observation_end DATE,
    last_updated TIMESTAMP,
    popularity INTEGER,
    notes TEXT,
    discovered_at TIMESTAMP,
    source VARCHAR DEFAULT 'fred',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Backward-compatible schema upgrades (for existing deployments)
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS category VARCHAR;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS frequency VARCHAR;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS units VARCHAR;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS seasonal_adjustment VARCHAR;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS observation_start DATE;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS observation_end DATE;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS last_updated TIMESTAMP;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS popularity INTEGER;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS notes TEXT;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS discovered_at TIMESTAMP;
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'fred';
ALTER TABLE raw.fred_series_metadata ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_fred_series_metadata_category
    ON raw.fred_series_metadata(category);
