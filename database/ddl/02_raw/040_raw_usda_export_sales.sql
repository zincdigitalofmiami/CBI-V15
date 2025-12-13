-- Raw USDA Export Sales
-- Weekly export sales data

CREATE TABLE IF NOT EXISTS raw.usda_export_sales (
    report_date DATE NOT NULL,
    commodity VARCHAR NOT NULL,
    destination_country VARCHAR,
    net_sales_mt DOUBLE,
    exports_mt DOUBLE,
    outstanding_sales_mt DOUBLE,
    source VARCHAR DEFAULT 'usda_exports',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, commodity, destination_country)
);

-- Key focus: China soybean purchases
-- Signals for China bucket

CREATE INDEX IF NOT EXISTS idx_export_sales_country 
    ON raw.usda_export_sales(destination_country);

