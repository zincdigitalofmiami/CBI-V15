-- MotherDuck Schema: raw_staging
-- Per-run temporary tables: <source>_<bucket>_<run_id>
-- Pattern: raw_staging.{source}_{bucket}_{run_id}

-- NOTE: Tables in this schema are created dynamically per ingestion run
-- Example naming:
--   raw_staging.fred_fx_20251204_010000
--   raw_staging.databento_daily_20251204_010000
--   raw_staging.scrapecreators_trump_20251204_010000

-- Schema created, but tables created at runtime by ingestion scripts
-- All tables follow idempotent pattern:
--   1. Load to raw_staging with WRITE_TRUNCATE
--   2. MERGE into raw.* on primary key
--   3. Log to ops.ingestion_completion

