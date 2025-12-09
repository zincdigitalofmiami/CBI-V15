-- ============================================================================
-- NOAA Weather Data Tables
-- ============================================================================
-- Source: NOAA Climate Data Online (CDO) API
-- Coverage: 14 agricultural regions (Brazil, Argentina, US)
-- Frequency: Daily
-- ============================================================================

-- ============================================================================
-- Table: raw.noaa_weather_daily
-- ============================================================================
-- Daily weather observations for agricultural regions
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.noaa_weather_daily (
    observation_date DATE NOT NULL,
    region_code TEXT NOT NULL,
    region_name TEXT NOT NULL,
    country TEXT NOT NULL,
    
    -- Temperature (Celsius)
    temp_max DOUBLE,
    temp_min DOUBLE,
    temp_avg DOUBLE,
    
    -- Precipitation (mm)
    precip_mm DOUBLE,
    
    -- Soil Moisture (if available)
    soil_moisture_pct DOUBLE,
    
    -- Growing Degree Days (GDD)
    gdd_base_10c DOUBLE,  -- Base 10°C (for soybeans)
    gdd_base_8c DOUBLE,   -- Base 8°C (for corn)
    
    -- Drought Indices (if available)
    palmer_drought_index DOUBLE,
    
    -- Data quality
    data_source TEXT,
    quality_flag TEXT,
    
    -- Metadata
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (observation_date, region_code)
);

COMMENT ON TABLE raw.noaa_weather_daily IS 'Daily weather observations for 14 agricultural regions';

-- ============================================================================
-- Table: raw.weather_regions
-- ============================================================================
-- Reference table for weather regions
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.weather_regions (
    region_code TEXT PRIMARY KEY,
    region_name TEXT NOT NULL,
    country TEXT NOT NULL,
    state_province TEXT,
    latitude DOUBLE,
    longitude DOUBLE,
    crop_type TEXT,  -- Primary crop (soybeans, corn, etc.)
    
    -- NOAA station IDs (if applicable)
    noaa_station_ids TEXT[],
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE raw.weather_regions IS 'Reference table for 14 agricultural weather regions';

-- ============================================================================
-- Insert Weather Regions
-- ============================================================================

INSERT INTO raw.weather_regions (region_code, region_name, country, state_province, crop_type) VALUES
-- Brazil (6 regions)
('BR_MT', 'Mato Grosso', 'Brazil', 'Mato Grosso', 'Soybeans'),
('BR_GO', 'Goiás', 'Brazil', 'Goiás', 'Soybeans'),
('BR_MS', 'Mato Grosso do Sul', 'Brazil', 'Mato Grosso do Sul', 'Soybeans'),
('BR_PR', 'Paraná', 'Brazil', 'Paraná', 'Soybeans'),
('BR_RS', 'Rio Grande do Sul', 'Brazil', 'Rio Grande do Sul', 'Soybeans'),
('BR_BA', 'Bahia', 'Brazil', 'Bahia', 'Soybeans'),

-- Argentina (4 regions)
('AR_BA', 'Buenos Aires', 'Argentina', 'Buenos Aires', 'Soybeans'),
('AR_CO', 'Córdoba', 'Argentina', 'Córdoba', 'Soybeans'),
('AR_SF', 'Santa Fe', 'Argentina', 'Santa Fe', 'Soybeans'),
('AR_ER', 'Entre Ríos', 'Argentina', 'Entre Ríos', 'Soybeans'),

-- United States (4 regions)
('US_ECB', 'Eastern Corn Belt', 'United States', 'IL/IN/OH', 'Corn/Soybeans'),
('US_WCB', 'Western Corn Belt', 'United States', 'IA/MN/NE', 'Corn/Soybeans'),
('US_NP', 'Northern Plains', 'United States', 'ND/SD', 'Wheat/Soybeans'),
('US_CP', 'Central Plains', 'United States', 'KS/NE', 'Wheat/Corn')
ON CONFLICT (region_code) DO NOTHING;

-- ============================================================================
-- Indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_noaa_weather_date 
ON raw.noaa_weather_daily(observation_date DESC);

CREATE INDEX IF NOT EXISTS idx_noaa_weather_region 
ON raw.noaa_weather_daily(region_code, observation_date DESC);

CREATE INDEX IF NOT EXISTS idx_noaa_weather_country 
ON raw.noaa_weather_daily(country, observation_date DESC);

-- ============================================================================
-- Example Queries
-- ============================================================================

-- Get latest weather for all regions
-- SELECT 
--     w.observation_date,
--     r.region_name,
--     r.country,
--     w.temp_avg,
--     w.precip_mm,
--     w.gdd_base_10c
-- FROM raw.noaa_weather_daily w
-- JOIN raw.weather_regions r ON w.region_code = r.region_code
-- WHERE w.observation_date = (SELECT MAX(observation_date) FROM raw.noaa_weather_daily)
-- ORDER BY r.country, r.region_name;

-- Get 30-day precipitation totals for Brazil soybean regions
-- SELECT 
--     r.region_name,
--     SUM(w.precip_mm) AS precip_30d,
--     AVG(w.temp_avg) AS avg_temp_30d
-- FROM raw.noaa_weather_daily w
-- JOIN raw.weather_regions r ON w.region_code = r.region_code
-- WHERE r.country = 'Brazil'
--   AND r.crop_type = 'Soybeans'
--   AND w.observation_date >= CURRENT_DATE - INTERVAL 30 DAY
-- GROUP BY r.region_name
-- ORDER BY precip_30d DESC;

-- Calculate drought stress (low precip + high temp)
-- SELECT 
--     observation_date,
--     region_code,
--     precip_mm,
--     temp_avg,
--     CASE 
--         WHEN precip_mm < 5 AND temp_avg > 30 THEN 'High Stress'
--         WHEN precip_mm < 10 AND temp_avg > 28 THEN 'Moderate Stress'
--         ELSE 'Normal'
--     END AS drought_stress
-- FROM raw.noaa_weather_daily
-- WHERE region_code LIKE 'BR_%'
--   AND observation_date >= CURRENT_DATE - INTERVAL 7 DAY
-- ORDER BY observation_date DESC, region_code;

