-- EPA RIN Prices (RFS compliance data)
CREATE TABLE IF NOT EXISTS raw.epa_rin_prices (
    date        DATE NOT NULL,
    series_id   TEXT NOT NULL,
    value       DOUBLE,
    created_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, series_id)
);

CREATE INDEX IF NOT EXISTS idx_epa_rin_series ON raw.epa_rin_prices (series_id);

