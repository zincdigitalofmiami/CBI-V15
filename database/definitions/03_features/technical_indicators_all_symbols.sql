-- ============================================================================
-- TECHNICAL INDICATORS TABLE: ALL 30+ SYMBOLS
-- ============================================================================
-- Stores pre-computed technical indicators for all futures symbols
-- Updated daily via ingestion pipeline
-- ============================================================================

CREATE TABLE IF NOT EXISTS features.technical_indicators_all_symbols (
    -- Keys
    as_of_date    DATE NOT NULL,
    symbol        TEXT NOT NULL,
    
    -- Price & Basic Features
    close         DOUBLE,
    lag_close_1d  DOUBLE,
    lag_close_5d  DOUBLE,
    lag_close_21d DOUBLE,
    
    -- Returns
    log_ret_1d    DOUBLE,
    log_ret_5d    DOUBLE,
    log_ret_21d   DOUBLE,
    
    -- Moving Averages
    sma_5         DOUBLE,
    sma_10        DOUBLE,
    sma_21        DOUBLE,
    sma_50        DOUBLE,
    sma_200       DOUBLE,
    
    -- Volatility
    volatility_21d       DOUBLE,
    
    -- RSI
    rsi_14        DOUBLE,
    
    -- MACD
    macd          DOUBLE,
    macd_signal   DOUBLE,
    macd_histogram DOUBLE,
    
    -- Bollinger Bands
    bb_upper      DOUBLE,
    bb_middle     DOUBLE,
    bb_lower      DOUBLE,
    bb_position   DOUBLE,  -- Z-score
    bb_width_pct  DOUBLE,
    
    -- ATR
    atr_14        DOUBLE,
    tr_pct        DOUBLE,
    
    -- Stochastic
    stoch_k       DOUBLE,
    stoch_d       DOUBLE,
    
    -- Momentum
    roc_10d       DOUBLE,
    roc_21d       DOUBLE,
    roc_63d       DOUBLE,
    momentum_10d  DOUBLE,
    momentum_21d  DOUBLE,
    
    -- Volume
    volume        DOUBLE,
    avg_volume_21d DOUBLE,
    volume_ratio  DOUBLE,
    volume_zscore DOUBLE,
    obv           DOUBLE,
    
    PRIMARY KEY (as_of_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_tech_indicators_date ON features.technical_indicators_all_symbols (as_of_date);
CREATE INDEX IF NOT EXISTS idx_tech_indicators_symbol ON features.technical_indicators_all_symbols (symbol);

-- ============================================================================
-- CROSS-ASSET CORRELATIONS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS features.cross_asset_correlations (
    as_of_date    DATE NOT NULL PRIMARY KEY,
    
    -- ZL correlations
    corr_zl_zs_60d DOUBLE,
    corr_zl_zm_60d DOUBLE,
    corr_zl_cl_60d DOUBLE,
    corr_zl_ho_60d DOUBLE,
    corr_zl_hg_60d DOUBLE,
    corr_zl_dx_60d DOUBLE,
    
    -- CL correlations
    corr_cl_ho_60d DOUBLE,
    corr_cl_rb_60d DOUBLE,
    corr_cl_dx_60d DOUBLE,
    
    -- Metals correlations
    corr_hg_gc_60d DOUBLE,
    corr_hg_dx_60d DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_cross_asset_date ON features.cross_asset_correlations (as_of_date);

-- ============================================================================
-- FUNDAMENTAL SPREADS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS features.fundamental_spreads (
    as_of_date           DATE NOT NULL PRIMARY KEY,
    
    -- Soy complex spreads
    board_crush_spread   DOUBLE,
    oil_share_of_crush   DOUBLE,
    
    -- Energy spreads
    boho_spread          DOUBLE,  -- Soy Oil vs Heating Oil
    crack_spread         DOUBLE,  -- Refining margin
    
    -- Macro proxies
    china_copper_proxy   DOUBLE,  -- HG as China demand
    dollar_index         DOUBLE   -- DX
);

CREATE INDEX IF NOT EXISTS idx_spreads_date ON features.fundamental_spreads (as_of_date);

-- ============================================================================
-- BIG 8 BUCKET SCORES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS features.big8_bucket_scores (
    as_of_date          DATE NOT NULL PRIMARY KEY,
    
    -- Bucket Scores (0-100 scale, 50 = neutral)
    crush_bucket_score   DOUBLE,
    china_bucket_score   DOUBLE,
    fx_bucket_score      DOUBLE,
    fed_bucket_score     DOUBLE,
    tariff_bucket_score  DOUBLE,
    biofuel_bucket_score DOUBLE,
    energy_bucket_score  DOUBLE,
    volatility_bucket_score DOUBLE,
    
    -- Key Underlying Metrics
    board_crush          DOUBLE,
    china_pulse          DOUBLE,
    dollar_index         DOUBLE,
    yield_curve_slope    DOUBLE,
    tariff_activity      DOUBLE,
    rin_d4               DOUBLE,
    crude_price          DOUBLE,
    vix                  DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_bucket_scores_date ON features.big8_bucket_scores (as_of_date);

