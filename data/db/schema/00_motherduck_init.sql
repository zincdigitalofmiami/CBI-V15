-- MotherDuck Initialization Script
-- Run this FIRST to create database and install extensions

-- Connect to MotherDuck (requires MOTHERDUCK_TOKEN)
-- Connection string: md:usoil_intelligence?motherduck_token=${MOTHERDUCK_TOKEN}

-- Install AnoFox extensions FROM COMMUNITY
INSTALL anofox_forecast FROM community;
INSTALL anofox_tabular FROM community;
INSTALL anofox_statistics FROM community;

-- Load extensions
LOAD anofox_forecast;
LOAD anofox_tabular;
LOAD anofox_statistics;

-- Verify installation
SELECT 'anofox_forecast' AS extension, * FROM anofox_forecast_version();

-- Create all 8 schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS raw_staging;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS reference;
CREATE SCHEMA IF NOT EXISTS signals;
CREATE SCHEMA IF NOT EXISTS ops;

-- Note: archive schema created only if needed for legacy data
-- CREATE SCHEMA IF NOT EXISTS archive;

-- Show all schemas
SELECT schema_name FROM information_schema.schemata ORDER BY schema_name;

-- Ready for table creation (run 01-09 schema files next)

