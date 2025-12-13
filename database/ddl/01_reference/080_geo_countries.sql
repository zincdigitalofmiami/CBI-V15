-- Geographic Reference: Countries
-- ISO country codes and metadata for trade/weather data

CREATE TABLE IF NOT EXISTS reference.geo_countries (
    country_code VARCHAR(3) PRIMARY KEY,  -- ISO 3166-1 alpha-3
    country_name VARCHAR NOT NULL,
    region VARCHAR,  -- 'Americas', 'Asia', 'Europe', etc.
    is_major_soybean_producer BOOLEAN DEFAULT FALSE,
    is_major_soybean_importer BOOLEAN DEFAULT FALSE,
    currency_code VARCHAR(3)
);

-- Key countries for ZL forecasting
INSERT INTO reference.geo_countries VALUES
('USA', 'United States', 'Americas', TRUE, FALSE, 'USD'),
('BRA', 'Brazil', 'Americas', TRUE, TRUE, 'BRL'),
('ARG', 'Argentina', 'Americas', TRUE, FALSE, 'ARS'),
('CHN', 'China', 'Asia', FALSE, TRUE, 'CNY'),
('IDN', 'Indonesia', 'Asia', FALSE, FALSE, 'IDR'),
('MYS', 'Malaysia', 'Asia', FALSE, FALSE, 'MYR'),
('EUR', 'European Union', 'Europe', FALSE, TRUE, 'EUR')
ON CONFLICT DO NOTHING;

