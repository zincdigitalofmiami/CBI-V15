-- Features: Rolling Correlations and Betas
-- Cross-asset relationships

CREATE TABLE IF NOT EXISTS features.rolling_corr_beta (
    date DATE PRIMARY KEY,
    -- ZL correlations (90-day rolling)
    corr_zl_zs_90d DECIMAL(6, 4),
    corr_zl_zm_90d DECIMAL(6, 4),
    corr_zl_cl_90d DECIMAL(6, 4),
    corr_zl_ho_90d DECIMAL(6, 4),
    corr_zl_fcpo_90d DECIMAL(6, 4),
    corr_zl_dx_90d DECIMAL(6, 4),
    corr_zl_hg_90d DECIMAL(6, 4),
    -- ZL betas
    beta_zl_cl_90d DECIMAL(6, 4),
    beta_zl_dx_90d DECIMAL(6, 4),
    beta_zl_vix_90d DECIMAL(6, 4),
    -- Spread correlations
    corr_crush_spread_vix_90d DECIMAL(6, 4),
    -- Regime indicators
    corr_regime VARCHAR,  -- 'HIGH_CORR', 'NORMAL', 'DECORRELATED'
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 90-day rolling window, updated daily
-- Correlations for Big 8 bucket interactions

