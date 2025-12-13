-- Reference: Canonical Symbols
-- 33 futures symbols tracked by V15

CREATE TABLE IF NOT EXISTS reference.symbols (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    exchange VARCHAR,
    category VARCHAR,  -- agricultural, energy, metals, treasuries, fx
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Categories index for filtering
CREATE INDEX IF NOT EXISTS idx_symbols_category 
    ON reference.symbols(category);

-- Primary symbol flag (ZL is primary for soybean oil forecasting)
CREATE INDEX IF NOT EXISTS idx_symbols_primary 
    ON reference.symbols(is_primary) WHERE is_primary = TRUE;

