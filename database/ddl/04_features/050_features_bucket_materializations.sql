-- Features: Bucket Materializations
-- Snapshot of features_dev bucket views for training

CREATE TABLE IF NOT EXISTS features.bucket_crush (
    as_of_date DATE PRIMARY KEY,
    crush_spread DECIMAL(10, 4),
    oil_share DECIMAL(5, 4),
    board_crush DECIMAL(10, 4),
    zl_zs_ratio DECIMAL(6, 4),
    zm_zs_ratio DECIMAL(6, 4),
    crush_momentum_21d DECIMAL(10, 6),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features.bucket_china (
    as_of_date DATE PRIMARY KEY,
    hg_zs_corr_90d DECIMAL(6, 4),
    export_sales_china_mt DOUBLE,
    export_sales_momentum DECIMAL(10, 6),
    china_demand_index DECIMAL(5, 2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features.bucket_fx (
    as_of_date DATE PRIMARY KEY,
    dxy DECIMAL(10, 4),
    brl_usd DECIMAL(10, 4),
    cny_usd DECIMAL(10, 4),
    dxy_momentum_21d DECIMAL(10, 6),
    brl_momentum_21d DECIMAL(10, 6),
    fx_volatility DECIMAL(10, 6),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features.bucket_fed (
    as_of_date DATE PRIMARY KEY,
    fed_funds DECIMAL(6, 4),
    yield_curve_10y2y DECIMAL(6, 4),
    nfci DECIMAL(10, 6),
    rate_momentum_63d DECIMAL(10, 6),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features.bucket_tariff (
    as_of_date DATE PRIMARY KEY,
    trump_sentiment DECIMAL(5, 4),
    policy_risk_score DECIMAL(5, 4),
    tariff_headline_count INT,
    trade_tension_index DECIMAL(5, 2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features.bucket_biofuel (
    as_of_date DATE PRIMARY KEY,
    rin_d4 DECIMAL(10, 4),
    rin_d6 DECIMAL(10, 4),
    boho_spread DECIMAL(10, 4),
    biodiesel_prod DOUBLE,
    rfs_compliance_pct DECIMAL(5, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features.bucket_energy (
    as_of_date DATE PRIMARY KEY,
    cl_close DECIMAL(10, 2),
    ho_close DECIMAL(10, 2),
    rb_close DECIMAL(10, 2),
    crack_spread_321 DECIMAL(10, 4),
    energy_zl_corr_90d DECIMAL(6, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features.bucket_volatility (
    as_of_date DATE PRIMARY KEY,
    vix DECIMAL(6, 2),
    realized_volatility_21d DECIMAL(10, 6),
    realized_volatility_63d DECIMAL(10, 6),
    stlfsi4 DECIMAL(10, 6),
    volatility_regime VARCHAR,  -- 'LOW', 'NORMAL', 'HIGH', 'CRISIS'
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

