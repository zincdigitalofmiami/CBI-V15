-- Staging: FRED Macro Clean
-- Wide-format macro panel with forward-filled values

CREATE TABLE IF NOT EXISTS staging.fred_macro_clean (
    date DATE PRIMARY KEY,
    -- Rates
    fed_funds DECIMAL(6, 4),
    treasury_10y DECIMAL(6, 4),
    treasury_2y DECIMAL(6, 4),
    yield_curve_10y2y DECIMAL(6, 4),
    -- FX
    dxy DECIMAL(10, 4),
    brl_usd DECIMAL(10, 4),
    cny_usd DECIMAL(10, 4),
    mxn_usd DECIMAL(10, 4),
    -- Volatility/Stress
    vix DECIMAL(6, 2),
    stlfsi4 DECIMAL(10, 6),
    nfci DECIMAL(10, 6),
    -- Inflation
    cpi_yoy DECIMAL(6, 4),
    pce_yoy DECIMAL(6, 4),
    -- Commodities
    wti_crude DECIMAL(10, 2),
    -- (60+ series total)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- All values forward-filled on non-release days
-- Point-in-time safe (no look-ahead)

