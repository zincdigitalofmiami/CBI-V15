-- Forecasts: Monte Carlo Scenarios
-- Probabilistic scenarios for risk analysis

CREATE TABLE IF NOT EXISTS forecasts.monte_carlo_scenarios (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,
    -- Probability distribution
    percentile_5 DECIMAL(10, 4),
    percentile_10 DECIMAL(10, 4),
    percentile_25 DECIMAL(10, 4),
    percentile_50 DECIMAL(10, 4),
    percentile_75 DECIMAL(10, 4),
    percentile_90 DECIMAL(10, 4),
    percentile_95 DECIMAL(10, 4),
    -- Risk metrics
    var_95 DECIMAL(10, 4),  -- Value at Risk (95%)
    cvar_95 DECIMAL(10, 4),  -- Conditional VaR (Expected Shortfall)
    var_99 DECIMAL(10, 4),
    cvar_99 DECIMAL(10, 4),
    -- Distribution stats
    mean DECIMAL(10, 4),
    std DECIMAL(10, 4),
    skewness DECIMAL(6, 4),
    kurtosis DECIMAL(6, 4),
    -- Simulation metadata
    n_simulations INT DEFAULT 10000,
    regime VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code)
);

-- Monte Carlo uses residual banks by horizon × regime
-- 10k-50k simulations per forecast

