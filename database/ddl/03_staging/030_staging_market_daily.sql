-- Staging: Market Daily Panel
-- Wide-format multi-symbol panel for feature engineering

CREATE TABLE IF NOT EXISTS staging.market_daily (
    date DATE PRIMARY KEY,
    -- ZL (primary)
    zl_close DECIMAL(10, 2),
    zl_volume BIGINT,
    zl_open_interest BIGINT,
    zl_return DECIMAL(10, 6),
    -- Crush complex
    zs_close DECIMAL(10, 2),
    zm_close DECIMAL(10, 2),
    -- Energy
    cl_close DECIMAL(10, 2),
    ho_close DECIMAL(10, 2),
    rb_close DECIMAL(10, 2),
    -- FX
    dx_close DECIMAL(10, 4),
    -- Palm
    fcpo_close DECIMAL(10, 2),
    -- Metals
    hg_close DECIMAL(10, 4),
    gc_close DECIMAL(10, 2),
    -- Volatility
    vix_close DECIMAL(6, 2),
    -- (Other symbols as needed)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily grain for all feature computations

