-- Geographic Reference: Administrative Regions
-- US states, Brazilian estados, etc. for granular weather/production data

CREATE TABLE IF NOT EXISTS reference.geo_admin_regions (
    region_id VARCHAR PRIMARY KEY,
    country_code VARCHAR(3) REFERENCES reference.geo_countries(country_code),
    region_name VARCHAR NOT NULL,
    region_type VARCHAR,  -- 'state', 'province', 'estado'
    is_major_crop_region BOOLEAN DEFAULT FALSE,
    primary_crop VARCHAR  -- 'soybeans', 'corn', 'wheat'
);

-- Key US states for soybean production
INSERT INTO reference.geo_admin_regions VALUES
('US-IA', 'USA', 'Iowa', 'state', TRUE, 'soybeans'),
('US-IL', 'USA', 'Illinois', 'state', TRUE, 'soybeans'),
('US-MN', 'USA', 'Minnesota', 'state', TRUE, 'soybeans'),
('US-IN', 'USA', 'Indiana', 'state', TRUE, 'soybeans'),
('US-NE', 'USA', 'Nebraska', 'state', TRUE, 'soybeans'),
('BR-MT', 'BRA', 'Mato Grosso', 'estado', TRUE, 'soybeans'),
('BR-PR', 'BRA', 'Paraná', 'estado', TRUE, 'soybeans'),
('BR-RS', 'BRA', 'Rio Grande do Sul', 'estado', TRUE, 'soybeans'),
('AR-BA', 'ARG', 'Buenos Aires', 'province', TRUE, 'soybeans'),
('AR-SF', 'ARG', 'Santa Fe', 'province', TRUE, 'soybeans')
ON CONFLICT DO NOTHING;

