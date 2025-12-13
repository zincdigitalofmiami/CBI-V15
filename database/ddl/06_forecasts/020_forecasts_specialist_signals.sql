-- Forecasts: Specialist Signals
-- Latest Big 8 signals for dashboard

CREATE TABLE IF NOT EXISTS forecasts.specialist_signals (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,
    -- Big 8 signals (for dashboard display)
    crush_signal DECIMAL(6, 4),
    crush_score DECIMAL(5, 2),  -- 0-100
    crush_direction INT,
    china_signal DECIMAL(6, 4),
    china_score DECIMAL(5, 2),
    china_direction INT,
    fx_signal DECIMAL(6, 4),
    fx_score DECIMAL(5, 2),
    fx_direction INT,
    fed_signal DECIMAL(6, 4),
    fed_score DECIMAL(5, 2),
    fed_direction INT,
    tariff_signal DECIMAL(6, 4),
    tariff_score DECIMAL(5, 2),
    tariff_direction INT,
    biofuel_signal DECIMAL(6, 4),
    biofuel_score DECIMAL(5, 2),
    biofuel_direction INT,
    energy_signal DECIMAL(6, 4),
    energy_score DECIMAL(5, 2),
    energy_direction INT,
    volatility_signal DECIMAL(6, 4),
    volatility_score DECIMAL(5, 2),
    volatility_direction INT,
    -- Aggregate
    composite_score DECIMAL(5, 2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code)
);

-- Dashboard shows these 8 bucket scores in real-time

