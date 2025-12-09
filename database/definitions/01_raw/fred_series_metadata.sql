-- FRED Series Metadata
-- Stores discovered FRED economic series from seed harvest
-- Categories: FX, Rates, Macro, Credit, Financial Conditions

CREATE TABLE IF NOT EXISTS raw.fred_series_metadata (
    -- Primary Key
    series_id TEXT PRIMARY KEY,
    
    -- Series Information
    title TEXT,
    category TEXT,  -- fx, rates, macro, credit, financial_conditions
    
    -- Data Characteristics
    frequency TEXT,  -- Daily, Weekly, Monthly, Quarterly, Annual
    units TEXT,  -- Percent, Index, Dollars, etc.
    seasonal_adjustment TEXT,  -- Seasonally Adjusted, Not Seasonally Adjusted
    
    -- Date Range
    observation_start DATE,
    observation_end DATE,
    
    -- Metadata
    last_updated TIMESTAMP,
    popularity INTEGER,  -- FRED popularity score
    notes TEXT,
    
    -- Discovery Tracking
    discovered_at TIMESTAMP DEFAULT current_timestamp,
    is_active BOOLEAN DEFAULT true,  -- Flag for series we actively ingest
    
    -- Ingestion Status
    last_ingested_at TIMESTAMP,
    observation_count INTEGER
);

-- Index for category queries
CREATE INDEX IF NOT EXISTS idx_fred_series_category ON raw.fred_series_metadata(category, is_active);

-- Index for popularity ranking
CREATE INDEX IF NOT EXISTS idx_fred_series_popularity ON raw.fred_series_metadata(popularity DESC);

