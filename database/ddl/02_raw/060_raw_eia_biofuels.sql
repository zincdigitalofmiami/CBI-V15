-- Raw EIA Biofuels Data
-- Energy Information Administration biofuels production/consumption

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

-- Key series for Biofuel bucket:
-- Biodiesel production by PADD
-- Renewable diesel production
-- Biofuel consumption
-- RFS mandate compliance

CREATE INDEX IF NOT EXISTS idx_eia_biofuels_series 
    ON raw.eia_biofuels(series_id);

