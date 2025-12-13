-- Raw USDA WASDE Reports
-- World Agricultural Supply and Demand Estimates

CREATE TABLE IF NOT EXISTS raw.usda_wasde (
    report_date DATE NOT NULL,
    commodity VARCHAR NOT NULL,
    country VARCHAR,
    metric VARCHAR NOT NULL,  -- 'production', 'exports', 'ending_stocks', etc.
    value DOUBLE,
    unit VARCHAR,
    source VARCHAR DEFAULT 'usda_wasde',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, commodity, country, metric)
);

-- Key metrics for ZL:
-- Soybean oil: production, crush, exports, ending_stocks
-- Soybeans: production, crush, exports
-- Palm oil: production (Indonesia, Malaysia)

CREATE INDEX IF NOT EXISTS idx_wasde_commodity 
    ON raw.usda_wasde(commodity);

