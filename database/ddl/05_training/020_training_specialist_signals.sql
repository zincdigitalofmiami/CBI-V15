-- Training: Specialist Signals
-- Standardized L0 outputs for meta-model

CREATE TABLE IF NOT EXISTS training.specialist_signals (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,
    -- Big 8 specialist signals (standardized)
    crush_signal DECIMAL(6, 4),
    china_signal DECIMAL(6, 4),
    fx_signal DECIMAL(6, 4),
    fed_signal DECIMAL(6, 4),
    tariff_signal DECIMAL(6, 4),
    biofuel_signal DECIMAL(6, 4),
    energy_signal DECIMAL(6, 4),
    volatility_signal DECIMAL(6, 4),
    -- Confidence weights
    crush_confidence DECIMAL(5, 4),
    china_confidence DECIMAL(5, 4),
    fx_confidence DECIMAL(5, 4),
    fed_confidence DECIMAL(5, 4),
    tariff_confidence DECIMAL(5, 4),
    biofuel_confidence DECIMAL(5, 4),
    energy_confidence DECIMAL(5, 4),
    volatility_confidence DECIMAL(5, 4),
    -- Aggregate
    mean_signal DECIMAL(6, 4),
    signal_dispersion DECIMAL(5, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code)
);

-- Signals standardized to [-1, 1] range
-- Used as features for L2 meta model

