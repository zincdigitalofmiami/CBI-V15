-- Weather Location Registry
-- Maps weather stations to crop regions

CREATE TABLE IF NOT EXISTS reference.weather_location_registry (
    location_id VARCHAR PRIMARY KEY,
    station_id VARCHAR,  -- NOAA station ID
    region_id VARCHAR REFERENCES reference.geo_admin_regions(region_id),
    latitude DECIMAL(9, 6),
    longitude DECIMAL(9, 6),
    elevation_m INT,
    location_name VARCHAR,
    is_active BOOLEAN DEFAULT TRUE
);

-- Key weather stations for crop monitoring
INSERT INTO reference.weather_location_registry VALUES
('US-IA-DES', 'USW00014933', 'US-IA', 41.534, -93.663, 294, 'Des Moines, IA', TRUE),
('US-IL-CHI', 'USW00094846', 'US-IL', 41.961, -87.907, 204, 'Chicago O''Hare, IL', TRUE),
('BR-MT-CUI', 'SBCY', 'BR-MT', -15.653, -56.106, 182, 'Cuiabá, MT', TRUE),
('AR-BA-EZE', 'SAEZ', 'AR-BA', -34.822, -58.536, 20, 'Buenos Aires, AR', TRUE)
ON CONFLICT DO NOTHING;

