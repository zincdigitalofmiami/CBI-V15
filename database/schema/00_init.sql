-- MotherDuck Initialization Script
-- Run this FIRST to create database and install extensions

-- Install AnoFox extensions FROM COMMUNITY (if available) or standard DuckDB extensions
INSTALL httpfs;
LOAD httpfs;

INSTALL parquet;
LOAD parquet;

-- Create all 8 schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS forecasts;
CREATE SCHEMA IF NOT EXISTS reference;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS tsci;

-- Verify schemas
SELECT schema_name FROM information_schema.schemata ORDER BY schema_name;
