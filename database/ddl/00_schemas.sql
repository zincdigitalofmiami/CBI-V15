-- MotherDuck/DuckDB Schema Initialization
-- Run this FIRST to create database schemas
--
-- Connection: md:usoil_intelligence?motherduck_token=${MOTHERDUCK_TOKEN}
-- Local: ATTACH 'md:usoil_intelligence' (no alias - use usoil_intelligence.schema.table)
--
-- V15.1 Schema Layout (9 schemas):
--   raw          - Immutable ingested data from collectors
--   staging      - Cleaned, normalized, typed data
--   features     - Production features + daily_ml_matrix
--   features_dev - Dev views (macro outputs pre-snapshot)
--   training     - OOF predictions, meta matrices, ensemble weights
--   forecasts    - Serving contract (dashboard reads this)
--   reference    - Symbols, calendars, regimes, driver maps
--   ops          - Run logs, ingestion completion, locks, alerts
--   explanations - Offline SHAP/feature-importance (weekly)

-- Create all 9 schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS features_dev;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS forecasts;
CREATE SCHEMA IF NOT EXISTS reference;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS explanations;
