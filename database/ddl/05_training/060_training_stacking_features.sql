-- Training: Stacking Features
-- OOF stacker training frame

CREATE TABLE IF NOT EXISTS training.stacking_features (
    as_of_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,
    fold_id INT NOT NULL,  -- Cross-validation fold
    -- OOF predictions from each model
    oof_crush DECIMAL(10, 6),
    oof_china DECIMAL(10, 6),
    oof_fx DECIMAL(10, 6),
    oof_fed DECIMAL(10, 6),
    oof_tariff DECIMAL(10, 6),
    oof_biofuel DECIMAL(10, 6),
    oof_energy DECIMAL(10, 6),
    oof_volatility DECIMAL(10, 6),
    oof_core_ts DECIMAL(10, 6),
    -- Target
    actual_return DECIMAL(10, 6),
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, horizon_code, fold_id)
);

-- OOF = Out-of-Fold predictions
-- Used to train the stacking/meta layer without leakage

