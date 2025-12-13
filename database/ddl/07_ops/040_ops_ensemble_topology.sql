-- Ops: Ensemble Topology
-- Current ensemble configuration snapshot

CREATE TABLE IF NOT EXISTS ops.ensemble_topology (
    snapshot_date DATE NOT NULL,
    horizon_code VARCHAR NOT NULL,
    regime VARCHAR NOT NULL,
    -- Topology JSON (full structure)
    topology JSON NOT NULL,
    -- Summary stats
    n_active_models INT,
    top_model VARCHAR,
    top_model_weight DECIMAL(5, 4),
    -- Changes
    models_added VARCHAR[],
    models_removed VARCHAR[],
    weight_changes JSON,
    -- Status
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (snapshot_date, horizon_code, regime)
);

-- Track ensemble changes over time for auditability

