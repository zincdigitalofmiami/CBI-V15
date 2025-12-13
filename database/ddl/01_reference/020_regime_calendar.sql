-- Regime Calendar Reference Table
-- Historical regime assignments (CALM, STRESSED, CRISIS)

CREATE TABLE IF NOT EXISTS reference.regime_calendar (
    date DATE PRIMARY KEY,
    regime VARCHAR NOT NULL,  -- 'CALM', 'STRESSED', 'CRISIS'
    volatility_zscore DECIMAL(6, 4),
    garch_variance DECIMAL(10, 6),
    vix_level DECIMAL(6, 2),
    regime_confidence DECIMAL(5, 4),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for regime filtering
CREATE INDEX IF NOT EXISTS idx_regime_calendar_regime 
    ON reference.regime_calendar(regime);

