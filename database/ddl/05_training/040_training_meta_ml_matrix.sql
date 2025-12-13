-- Training: Meta ML Matrix
-- L2 meta-model input features

CREATE TABLE IF NOT EXISTS training.meta_ml_matrix (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,
    -- L0 specialist signals (from training.specialist_signals)
    crush_signal DECIMAL(6, 4),
    china_signal DECIMAL(6, 4),
    fx_signal DECIMAL(6, 4),
    fed_signal DECIMAL(6, 4),
    tariff_signal DECIMAL(6, 4),
    biofuel_signal DECIMAL(6, 4),
    energy_signal DECIMAL(6, 4),
    volatility_signal DECIMAL(6, 4),
    -- L1 core TS quantiles
    core_q10 DECIMAL(10, 4),
    core_q50 DECIMAL(10, 4),
    core_q90 DECIMAL(10, 4),
    -- Regime context
    volatility_regime VARCHAR,
    regime_confidence DECIMAL(5, 4),
    -- Signal agreement metrics
    specialist_agreement DECIMAL(5, 4),  -- How many specialists agree on direction
    core_specialist_alignment DECIMAL(5, 4),  -- Core vs specialist alignment
    -- Target (for training)
    actual_return DECIMAL(10, 6),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code)
);

-- Meta model learns optimal combination of L0 + L1 outputs

