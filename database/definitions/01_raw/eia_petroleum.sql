-- EIA Petroleum & Biofuels Data for Procurement Intelligence
-- 
-- Contains data critical for soybean oil procurement decisions:
--   - Biofuel feedstock inputs (soybean oil, corn oil, tallow, yellow grease)
--   - Diesel/heating oil spot prices
--   - Refinery utilization rates
--   - Petroleum stocks
--
-- Data sources:
--   - https://api.eia.gov/v2/petroleum/pnp/feedbiofuel (biofuel feedstocks)
--   - https://api.eia.gov/v2/petroleum/pri/spt (spot prices)
--   - https://api.eia.gov/v2/petroleum/pnp/wiup (refinery utilization)

CREATE TABLE IF NOT EXISTS raw.eia_petroleum (
    date          DATE NOT NULL,
    series_id     TEXT NOT NULL,
    value         DOUBLE,
    category      TEXT,           -- biofuel_feedstock, spot_price, refinery, stocks
    units         TEXT,           -- MMLB, $/GAL, %, MBBL
    created_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, series_id)
);

-- Index for category-based queries
CREATE INDEX IF NOT EXISTS idx_eia_petroleum_category ON raw.eia_petroleum (category, date);

-- Index for series lookups
CREATE INDEX IF NOT EXISTS idx_eia_petroleum_series ON raw.eia_petroleum (series_id);

-- =============================================================================
-- KEY SERIES REFERENCE
-- =============================================================================
-- 
-- BIOFUEL FEEDSTOCKS (Monthly, MMLB = Million Pounds):
--   M_EPOOBDSOR_YIFBP_NUS_MMLB  - Soybean Oil inputs to Biodiesel Production
--   M_EPOOBDSOR_YIFRD_NUS_MMLB  - Soybean Oil inputs to Renewable Diesel
--   M_EPOOBDCNOR_YIFBP_NUS_MMLB - Corn Oil inputs to Biodiesel (competitor)
--   M_EPOOBD4OR_YIFBP_NUS_MMLB  - Yellow Grease inputs (competitor)
--   M_EPOOBD5OR_YIFBP_NUS_MMLB  - Tallow inputs (competitor)
--
-- SPOT PRICES (Weekly, $/GAL):
--   EER_EPD2DXL0_PF4_RGC_DPG    - Gulf Coast ULSD Diesel
--   EER_EPD2DXL0_PF4_Y35NY_DPG  - NY Harbor ULSD Diesel
--   EER_EPD2F_PF4_Y35NY_DPG     - NY Harbor Heating Oil
--   RWTC                        - WTI Crude Oil ($/BBL)
--
-- REFINERY (Weekly):
--   WPULEUS3                    - US Refinery Utilization %
--   WGFUPUS2                    - Gross Inputs to Refineries (MBBL/D)

