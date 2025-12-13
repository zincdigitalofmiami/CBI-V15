-- V0001: Initialize schemas
-- Creates all 9 schemas for V15.1

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS features_dev;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS forecasts;
CREATE SCHEMA IF NOT EXISTS reference;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS explanations;

-- Verify
SELECT schema_name FROM information_schema.schemata 
WHERE schema_name IN ('raw', 'staging', 'features', 'features_dev', 
                      'training', 'forecasts', 'reference', 'ops', 'explanations')
ORDER BY schema_name;

