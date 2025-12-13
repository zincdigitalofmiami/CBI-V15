-- Trading Calendar Reference Table
-- Market days, holidays, trading sessions

CREATE TABLE IF NOT EXISTS reference.trading_calendar (
    date DATE PRIMARY KEY,
    is_trading_day BOOLEAN NOT NULL DEFAULT TRUE,
    holiday_name VARCHAR,
    market VARCHAR DEFAULT 'CME',  -- CME, ICE, etc.
    session_type VARCHAR DEFAULT 'regular',  -- regular, half_day, closed
    notes TEXT
);

-- Index for efficient date range queries
CREATE INDEX IF NOT EXISTS idx_trading_calendar_trading_day 
    ON reference.trading_calendar(date) WHERE is_trading_day = TRUE;

