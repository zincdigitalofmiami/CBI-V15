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

