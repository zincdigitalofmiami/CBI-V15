-- Raw Weather Data
-- NOAA weather observations

CREATE TABLE IF NOT EXISTS raw.weather_noaa (
    station_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    tavg_c DECIMAL(5, 2),
    tmin_c DECIMAL(5, 2),
    tmax_c DECIMAL(5, 2),
    prcp_mm DECIMAL(6, 2),
    snow_mm DECIMAL(6, 2),
    region VARCHAR,
    country VARCHAR,
    source VARCHAR DEFAULT 'noaa',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (station_id, date)
);

-- Key regions: US Corn Belt, Brazil Mato Grosso, Argentina Pampas

CREATE INDEX IF NOT EXISTS idx_weather_region 
    ON raw.weather_noaa(region);
CREATE INDEX IF NOT EXISTS idx_weather_date 
    ON raw.weather_noaa(date);

