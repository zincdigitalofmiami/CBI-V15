-- FRED Economic Data (from fred/ ingestion scripts)
CREATE TABLE IF NOT EXISTS raw.fred_economic (
    date          DATE NOT NULL,
    series_id     TEXT NOT NULL,
    value         DOUBLE,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, series_id)
);

CREATE INDEX IF NOT EXISTS idx_fred_series ON raw.fred_economic (series_id);
