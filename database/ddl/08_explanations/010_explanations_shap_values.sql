-- Explanations: SHAP Values
-- Weekly computed feature importance

CREATE TABLE IF NOT EXISTS explanations.shap_values (
    as_of_date DATE NOT NULL,
    model_name VARCHAR NOT NULL,
    horizon_code VARCHAR NOT NULL,
    feature_name VARCHAR NOT NULL,
    -- SHAP values
    mean_abs_shap DECIMAL(10, 6),
    mean_shap DECIMAL(10, 6),  -- Can be positive or negative
    std_shap DECIMAL(10, 6),
    -- Ranking
    importance_rank INT,
    -- Direction
    positive_impact_ratio DECIMAL(5, 4),  -- % of times feature pushes up
    -- Metadata
    n_samples INT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, model_name, horizon_code, feature_name)
);

-- Computed weekly (expensive operation)
-- Dashboard reads for feature importance display

CREATE INDEX IF NOT EXISTS idx_shap_model 
    ON explanations.shap_values(model_name);
CREATE INDEX IF NOT EXISTS idx_shap_rank 
    ON explanations.shap_values(importance_rank);

