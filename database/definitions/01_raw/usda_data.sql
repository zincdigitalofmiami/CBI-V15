-- ============================================================================
-- USDA Data Tables
-- ============================================================================
-- Source: USDA NASS Quick Stats API, WASDE Reports
-- Coverage: 10 series (WASDE, export sales, crop progress/conditions)
-- Frequency: Monthly (WASDE), Weekly (export sales, crop progress)
-- ============================================================================

-- ============================================================================
-- Table: raw.usda_wasde
-- ============================================================================
-- World Agricultural Supply & Demand Estimates (WASDE)
-- Released monthly (usually 12th of each month)
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.usda_wasde (
    report_date DATE NOT NULL,
    commodity TEXT NOT NULL,
    country TEXT NOT NULL,
    metric TEXT NOT NULL,  -- 'production', 'consumption', 'exports', 'ending_stocks'
    
    -- Values (in metric tons or bushels)
    value DOUBLE,
    unit TEXT,  -- 'MT' (metric tons), 'bushels', etc.
    
    -- Forecast vs Actual
    is_forecast BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    report_month TEXT,  -- 'January 2025', etc.
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (report_date, commodity, country, metric)
);

COMMENT ON TABLE raw.usda_wasde IS 'USDA World Agricultural Supply & Demand Estimates (monthly)';

-- ============================================================================
-- Table: raw.usda_export_sales
-- ============================================================================
-- Weekly export sales reports
-- Released every Thursday at 8:30 AM ET
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.usda_export_sales (
    report_date DATE NOT NULL,
    commodity TEXT NOT NULL,
    destination_country TEXT NOT NULL,
    
    -- Sales (metric tons)
    net_sales_mt DOUBLE,
    accumulated_exports_mt DOUBLE,
    outstanding_sales_mt DOUBLE,
    
    -- Marketing year
    marketing_year TEXT,  -- '2024/2025'
    
    -- Metadata
    report_week_ending DATE,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (report_date, commodity, destination_country)
);

COMMENT ON TABLE raw.usda_export_sales IS 'USDA weekly export sales reports';

-- ============================================================================
-- Table: raw.usda_crop_progress
-- ============================================================================
-- Weekly crop progress and condition reports
-- Released every Monday at 4:00 PM ET
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.usda_crop_progress (
    report_date DATE NOT NULL,
    state TEXT NOT NULL,
    commodity TEXT NOT NULL,
    
    -- Progress (% of crop)
    planted_pct DOUBLE,
    emerged_pct DOUBLE,
    blooming_pct DOUBLE,
    setting_pods_pct DOUBLE,
    dropping_leaves_pct DOUBLE,
    harvested_pct DOUBLE,
    
    -- Condition (% of crop in each category)
    condition_very_poor_pct DOUBLE,
    condition_poor_pct DOUBLE,
    condition_fair_pct DOUBLE,
    condition_good_pct DOUBLE,
    condition_excellent_pct DOUBLE,
    
    -- Derived: Good + Excellent
    condition_good_excellent_pct DOUBLE,
    
    -- Metadata
    report_week_ending DATE,
    crop_year INTEGER,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (report_date, state, commodity)
);

COMMENT ON TABLE raw.usda_crop_progress IS 'USDA weekly crop progress and condition reports';

-- ============================================================================
-- Table: raw.usda_price_basis
-- ============================================================================
-- Price basis series (FOB, CIF prices)
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.usda_price_basis (
    observation_date DATE NOT NULL,
    commodity TEXT NOT NULL,
    location TEXT NOT NULL,  -- 'Brazil FOB Paranaguá', 'Argentina FOB Rosario', etc.
    price_type TEXT NOT NULL,  -- 'FOB', 'CIF'
    
    -- Price (USD per metric ton)
    price_usd_mt DOUBLE,
    
    -- Metadata
    data_source TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (observation_date, commodity, location, price_type)
);

COMMENT ON TABLE raw.usda_price_basis IS 'Price basis series (FOB, CIF) for soybeans and products';

-- ============================================================================
-- Indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_wasde_date ON raw.usda_wasde(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_wasde_commodity ON raw.usda_wasde(commodity, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_export_sales_date ON raw.usda_export_sales(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_export_sales_commodity ON raw.usda_export_sales(commodity, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_crop_progress_date ON raw.usda_crop_progress(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_crop_progress_commodity ON raw.usda_crop_progress(commodity, state, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_price_basis_date ON raw.usda_price_basis(observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_price_basis_commodity ON raw.usda_price_basis(commodity, location, observation_date DESC);

-- ============================================================================
-- Example Queries
-- ============================================================================

-- Get latest WASDE production estimates
-- SELECT 
--     commodity,
--     country,
--     value AS production_mt,
--     report_month
-- FROM raw.usda_wasde
-- WHERE metric = 'production'
--   AND report_date = (SELECT MAX(report_date) FROM raw.usda_wasde)
-- ORDER BY commodity, country;

-- Get weekly export sales to China
-- SELECT 
--     report_date,
--     commodity,
--     net_sales_mt,
--     accumulated_exports_mt
-- FROM raw.usda_export_sales
-- WHERE destination_country = 'China'
--   AND commodity IN ('Soybeans', 'Soybean Oil', 'Soybean Meal')
-- ORDER BY report_date DESC, commodity
-- LIMIT 20;

-- Get crop condition trends (Good + Excellent %)
-- SELECT 
--     report_date,
--     state,
--     commodity,
--     condition_good_excellent_pct
-- FROM raw.usda_crop_progress
-- WHERE commodity = 'Soybeans'
--   AND state IN ('Iowa', 'Illinois', 'Indiana')
--   AND report_date >= CURRENT_DATE - INTERVAL 90 DAY
-- ORDER BY report_date DESC, state;

