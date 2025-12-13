-- Raw EPA RIN Prices
-- Renewable Identification Numbers weekly prices

CREATE TABLE IF NOT EXISTS raw.epa_rin_prices (
    date DATE NOT NULL,
    rin_type VARCHAR NOT NULL,  -- 'D3' (cellulosic), 'D4' (biomass-based diesel), 'D5' (advanced), 'D6' (renewable fuel)
    price DECIMAL(10, 4),
    unit VARCHAR DEFAULT 'USD/gallon',
    source VARCHAR DEFAULT 'epa',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, rin_type)
);

-- D4 RINs most relevant for soybean oil (biodiesel feedstock)
-- Weekly data, FREE from EPA

CREATE INDEX IF NOT EXISTS idx_rin_prices_type 
    ON raw.epa_rin_prices(rin_type);

