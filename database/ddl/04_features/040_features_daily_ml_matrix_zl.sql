-- Features: Daily ML Matrix (ZL)
-- Wide matrix with all production features (~300 columns)

CREATE TABLE IF NOT EXISTS features.daily_ml_matrix_zl (
    as_of_date DATE PRIMARY KEY,
    
    -- === PRICE FEATURES (databento_*) ===
    databento_zl_close DECIMAL(10, 2),
    databento_zl_volume BIGINT,
    databento_zl_open_interest BIGINT,
    databento_zl_return_1d DECIMAL(10, 6),
    databento_zl_return_5d DECIMAL(10, 6),
    databento_zl_return_21d DECIMAL(10, 6),
    
    -- === TECHNICAL (tech_*) ===
    tech_zl_sma_20 DECIMAL(10, 2),
    tech_zl_sma_50 DECIMAL(10, 2),
    tech_zl_rsi_14 DECIMAL(5, 2),
    tech_zl_macd_histogram DECIMAL(10, 4),
    tech_zl_bb_pct DECIMAL(5, 4),
    tech_zl_atr_14 DECIMAL(10, 4),
    
    -- === VOLATILITY (volatility_*) ===
    volatility_zl_21d DECIMAL(10, 6),
    volatility_zl_63d DECIMAL(10, 6),
    volatility_vix DECIMAL(6, 2),
    volatility_stlfsi4 DECIMAL(10, 6),
    
    -- === CRUSH (crush_*) ===
    crush_spread DECIMAL(10, 4),
    crush_oil_share DECIMAL(5, 4),
    crush_board_crush DECIMAL(10, 4),
    
    -- === CHINA (china_*) ===
    china_hg_zs_corr_90d DECIMAL(6, 4),
    china_export_sales_net_mt DOUBLE,
    
    -- === FX (fx_*) ===
    fx_dxy DECIMAL(10, 4),
    fx_brl_usd DECIMAL(10, 4),
    fx_dxy_momentum_21d DECIMAL(10, 6),
    
    -- === FED (fed_*) ===
    fed_funds DECIMAL(6, 4),
    fed_yield_curve_10y2y DECIMAL(6, 4),
    fed_nfci DECIMAL(10, 6),
    
    -- === TARIFF (tariff_*) ===
    tariff_trump_sentiment DECIMAL(5, 4),
    tariff_policy_risk_score DECIMAL(5, 4),
    
    -- === BIOFUEL (biofuel_*) ===
    biofuel_rin_d4 DECIMAL(10, 4),
    biofuel_boho_spread DECIMAL(10, 4),
    
    -- === ENERGY (energy_*) ===
    energy_cl_close DECIMAL(10, 2),
    energy_ho_close DECIMAL(10, 2),
    energy_crack_spread DECIMAL(10, 4),
    
    -- === CFTC (cftc_*) ===
    cftc_managed_money_net BIGINT,
    cftc_managed_money_net_pctile DECIMAL(5, 2),
    
    -- (Total: ~300 features)
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- This is the CANONICAL feature matrix for training
-- All features point-in-time safe (no look-ahead)

