-- Crush data (from external sources)
CREATE TABLE IF NOT EXISTS staging.crush_daily (
    as_of_date              DATE PRIMARY KEY,
    nopa_volume_us          DOUBLE,
    capacity_utilization    DOUBLE
);
