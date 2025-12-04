-- AnoFox Feature Macros (Big 8 Logic)

-- 1. Base Keys
CREATE OR REPLACE MACRO ano_zl_base_keys() AS TABLE
SELECT
  d.as_of_date::DATE AS as_of_date,
  'ZL'::TEXT        AS symbol
FROM reference.trading_calendar d
WHERE d.is_trading_day = TRUE;

-- 2. Crush Features (Stub)
CREATE OR REPLACE MACRO ano_zl_crush_features() AS TABLE
SELECT
  c.as_of_date,
  c.nopa_volume_us              AS crush_nopa_volume_us_raw,
  c.capacity_utilization        AS crush_nopa_capacity_utilization_raw,
  -- Add more features here
FROM staging.crush_daily c;

-- 3. China Features (Stub)
CREATE OR REPLACE MACRO ano_zl_china_features() AS TABLE
SELECT
  c.as_of_date,
  c.soybean_imports_mmt         AS china_soybean_imports_mmt_raw,
  -- Add more features here
FROM staging.china_daily c;

-- ... Add macros for FX, Fed, Tariff, Biofuel, Energy, Vol ...

-- 9. Final Feature Matrix
CREATE OR REPLACE MACRO ano_zl_feature_matrix_v15() AS TABLE
WITH
  base AS ( SELECT * FROM ano_zl_base_keys() ),
  crush AS ( SELECT * FROM ano_zl_crush_features() ),
  china AS ( SELECT * FROM ano_zl_china_features() )
  -- ... other buckets
SELECT
  b.as_of_date,
  b.symbol,
  -- Join all features
  crush.* EXCLUDE(as_of_date),
  china.* EXCLUDE(as_of_date)
FROM base b
LEFT JOIN crush USING (as_of_date)
LEFT JOIN china USING (as_of_date);
