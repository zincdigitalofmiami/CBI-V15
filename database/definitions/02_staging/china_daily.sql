-- China demand data
CREATE TABLE IF NOT EXISTS staging.china_daily (
    as_of_date              DATE PRIMARY KEY,
    soybean_imports_mmt     DOUBLE
);
