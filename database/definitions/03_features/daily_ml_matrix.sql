-- ============================================================================
-- FEATURE TABLES: DENORMALIZED FEATURE MATRICES
-- ============================================================================
-- Main V15 Feature Matrix with 276+ features from all 30+ symbols
-- ============================================================================

-- Main Feature Matrix (Big 8 + Technical Indicators + Cross-Asset Features)
CREATE TABLE IF NOT EXISTS features.daily_ml_matrix_zl (
    -- ========================================================================
    -- KEYS & METADATA
    -- ========================================================================
    as_of_date    DATE NOT NULL,
    symbol        TEXT NOT NULL,
    regime        TEXT,

    -- ========================================================================
    -- PRICE & TECHNICAL INDICATORS (40+ features per symbol)
    -- ========================================================================
    close         DOUBLE,
    lag_close_1d  DOUBLE,
    lag_close_5d  DOUBLE,
    lag_close_21d DOUBLE,
    log_ret_1d    DOUBLE,
    log_ret_5d    DOUBLE,
    log_ret_21d   DOUBLE,
    sma_5         DOUBLE,
    sma_10        DOUBLE,
    sma_21        DOUBLE,
    sma_50        DOUBLE,
    sma_200       DOUBLE,
    volatility_21d DOUBLE,
    rsi_14        DOUBLE,
    macd          DOUBLE,
    macd_signal   DOUBLE,
    macd_histogram DOUBLE,
    bb_upper      DOUBLE,
    bb_middle     DOUBLE,
    bb_lower      DOUBLE,
    bb_position   DOUBLE,
    bb_width_pct  DOUBLE,
    atr_14        DOUBLE,
    tr_pct        DOUBLE,
    stoch_k       DOUBLE,
    stoch_d       DOUBLE,
    roc_10d       DOUBLE,
    roc_21d       DOUBLE,
    roc_63d       DOUBLE,
    momentum_10d  DOUBLE,
    momentum_21d  DOUBLE,
    volume        DOUBLE,
    avg_volume_21d DOUBLE,
    volume_ratio  DOUBLE,
    volume_zscore DOUBLE,
    obv           DOUBLE,

    -- ========================================================================
    -- CROSS-ASSET CORRELATIONS (11 features)
    -- ========================================================================
    corr_zl_zs_60d DOUBLE,
    corr_zl_zm_60d DOUBLE,
    corr_zl_cl_60d DOUBLE,
    corr_zl_ho_60d DOUBLE,
    corr_zl_hg_60d DOUBLE,
    corr_zl_dx_60d DOUBLE,
    corr_cl_ho_60d DOUBLE,
    corr_cl_rb_60d DOUBLE,
    corr_cl_dx_60d DOUBLE,
    corr_hg_gc_60d DOUBLE,
    corr_hg_dx_60d DOUBLE,

    -- ========================================================================
    -- FUNDAMENTAL SPREADS (6 features)
    -- ========================================================================
    board_crush_spread DOUBLE,
    oil_share_of_crush DOUBLE,
    boho_spread        DOUBLE,
    crack_spread       DOUBLE,
    china_copper_proxy DOUBLE,
    dollar_index       DOUBLE,

    -- ========================================================================
    -- BIG 8 BUCKET SCORES (16 features: 8 scores + 8 key metrics)
    -- ========================================================================
    crush_bucket_score   DOUBLE,
    china_bucket_score   DOUBLE,
    fx_bucket_score      DOUBLE,
    fed_bucket_score     DOUBLE,
    tariff_bucket_score  DOUBLE,
    biofuel_bucket_score DOUBLE,
    energy_bucket_score  DOUBLE,
    volatility_bucket_score DOUBLE,
    board_crush          DOUBLE,
    china_pulse          DOUBLE,
    yield_curve_slope    DOUBLE,
    tariff_activity      DOUBLE,
    rin_d4               DOUBLE,
    crude_price          DOUBLE,
    vix                  DOUBLE,

    -- ========================================================================
    -- BIG 8 NEURAL SCORES (8 features - populated by ML models)
    -- ========================================================================
    crush_neural_score   DOUBLE,
    china_neural_score   DOUBLE,
    fx_neural_score      DOUBLE,
    fed_neural_score     DOUBLE,
    tariff_neural_score  DOUBLE,
    biofuel_neural_score DOUBLE,
    energy_neural_score  DOUBLE,
    volatility_neural_score DOUBLE,

    -- ========================================================================
    -- MASTER NEURAL SCORE (1 feature - ensemble of bucket neural scores)
    -- ========================================================================
    master_neural_score  DOUBLE,

    -- ========================================================================
    -- TARGETS (8 features: 4 price + 4 return)
    -- ========================================================================
    target_price_1w DOUBLE,
    target_price_1m DOUBLE,
    target_price_3m DOUBLE,
    target_price_6m DOUBLE,
    target_ret_1w   DOUBLE,
    target_ret_1m   DOUBLE,
    target_ret_3m   DOUBLE,
    target_ret_6m   DOUBLE,

    PRIMARY KEY (as_of_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_features_date ON features.daily_ml_matrix_zl (as_of_date);
CREATE INDEX IF NOT EXISTS idx_features_symbol ON features.daily_ml_matrix_zl (symbol);
CREATE INDEX IF NOT EXISTS idx_features_regime ON features.daily_ml_matrix_zl (regime);
