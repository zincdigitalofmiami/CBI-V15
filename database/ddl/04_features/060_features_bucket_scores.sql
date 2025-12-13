-- Features: Bucket Scores
-- Dashboard-ready scores (0-100) for Big 8

CREATE TABLE IF NOT EXISTS features.bucket_scores (
    as_of_date DATE PRIMARY KEY,
    -- Big 8 scores (0-100, 50 = neutral)
    crush_score DECIMAL(5, 2),
    china_score DECIMAL(5, 2),
    fx_score DECIMAL(5, 2),
    fed_score DECIMAL(5, 2),
    tariff_score DECIMAL(5, 2),
    biofuel_score DECIMAL(5, 2),
    energy_score DECIMAL(5, 2),
    volatility_score DECIMAL(5, 2),
    -- Composite
    composite_score DECIMAL(5, 2),
    -- Direction signals
    crush_direction INT,  -- -1, 0, 1
    china_direction INT,
    fx_direction INT,
    fed_direction INT,
    tariff_direction INT,
    biofuel_direction INT,
    energy_direction INT,
    volatility_direction INT,
    -- Confidence
    score_confidence DECIMAL(5, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scores are for DASHBOARD display
-- Features (vectors) are for MODEL training
-- Different purposes, different tables

