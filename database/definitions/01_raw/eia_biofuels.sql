-- EIA Biofuels Data (from eia/ ingestion scripts)
CREATE TABLE IF NOT EXISTS raw.eia_biofuels (
    date          DATE NOT NULL,
    series_id     TEXT NOT NULL,
    value         DOUBLE,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, series_id)
);

CREATE INDEX IF NOT EXISTS idx_eia_series ON raw.eia_biofuels (series_id);
