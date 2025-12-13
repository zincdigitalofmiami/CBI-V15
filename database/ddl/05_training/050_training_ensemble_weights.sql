-- Training: Ensemble Weights
-- L3 WeightedEnsemble + L2.5 greedy overlay

CREATE TABLE IF NOT EXISTS training.ensemble_weights (
    horizon_code VARCHAR NOT NULL,
    regime VARCHAR NOT NULL,  -- 'CALM', 'STRESSED', 'CRISIS'
    -- Model weights
    model_name VARCHAR NOT NULL,
    model_tier VARCHAR NOT NULL,  -- 'specialist', 'core', 'meta'
    weight DECIMAL(5, 4) NOT NULL,
    -- Performance metrics
    oof_mape DECIMAL(10, 6),
    oof_directional_accuracy DECIMAL(5, 4),
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (horizon_code, regime, model_name)
);

-- Weights vary by regime
-- L2.5 greedy overlay applies stability/guardrail adjustments

CREATE INDEX IF NOT EXISTS idx_ensemble_weights_regime 
    ON training.ensemble_weights(regime);

