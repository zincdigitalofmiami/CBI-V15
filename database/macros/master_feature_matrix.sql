-- ============================================================================
-- MASTER FEATURE MATRIX: ALL 31 FUTURES SYMBOLS WITH 93+ FEATURES
-- ============================================================================
-- Combines:
-- - Technical indicators for all symbols
-- - Cross-asset correlations and spreads
-- - Big 8 bucket scores
-- - Macro/fundamental data
-- - Sentiment scores
-- ============================================================================

-- ============================================================================
-- SYMBOL LIST (31 Futures Contracts)
-- ============================================================================
-- Agricultural (8):
--   ZL (Soybean Oil), ZS (Soybeans), ZM (Soybean Meal), ZC (Corn),
--   ZW (Wheat), ZO (Oats), HE (Lean Hogs), FCPO (Palm Oil)
-- Energy (4):
--   CL (WTI Crude), HO (Heating Oil), RB (RBOB Gas), NG (Natural Gas)
-- Metals (5):
--   HG (Copper), GC (Gold), SI (Silver), PL (Platinum), PA (Palladium)
-- Treasuries (3):
--   ZF (5Y Note), ZN (10Y Note), ZB (30Y Bond)
-- FX Futures (10):
--   6E (Euro), 6J (Yen), 6B (Pound), 6C (CAD), 6A (AUD), 6N (NZD),
--   6M (MXN), 6L (BRL), 6S (CHF), DX (Dollar Index)
-- ============================================================================

-- ============================================================================
-- MACRO: Build Feature Matrix for Single Symbol
-- ============================================================================
CREATE OR REPLACE MACRO build_symbol_features(sym) AS TABLE
WITH
    -- Technical indicators for this symbol
    tech_indicators AS (
        SELECT * FROM calc_all_technical_indicators(sym)
    ),
    -- Targets for this symbol
    targets AS (
        SELECT * FROM feat_targets_block(sym)
    ),
    -- Cross-asset correlations (this symbol vs others)
    correlations AS (
        SELECT
            as_of_date,
            corr_zl_zs_60d,
            corr_zl_zm_60d,
            corr_zl_cl_60d,
            corr_zl_ho_60d,
            corr_zl_hg_60d,
            corr_zl_dx_60d,
            corr_cl_ho_60d,
            corr_cl_rb_60d,
            corr_cl_dx_60d,
            corr_hg_gc_60d,
            corr_hg_dx_60d
        FROM calc_correlation_matrix(60)
    ),
    -- Fundamental spreads
    spreads AS (
        SELECT
            as_of_date,
            board_crush_spread,
            oil_share_of_crush,
            boho_spread,
            crack_spread,
            china_copper_proxy,
            dollar_index
        FROM calc_fundamental_spreads()
    ),
    -- Big 8 bucket scores
    buckets AS (
        SELECT
            as_of_date,
            crush_bucket_score,
            china_bucket_score,
            fx_bucket_score,
            fed_bucket_score,
            tariff_bucket_score,
            biofuel_bucket_score,
            energy_bucket_score,
            volatility_bucket_score,
            board_crush,
            china_pulse,
            dollar_index AS dx_close,
            yield_curve_slope,
            tariff_activity,
            rin_d4,
            crude_price,
            vix
        FROM calc_all_bucket_scores()
    )
SELECT
    t.as_of_date,
    t.symbol,
    
    -- ========================================================================
    -- PRICE & TECHNICAL INDICATORS (40+ features)
    -- ========================================================================
    t.close,
    t.lag_close_1d, t.lag_close_5d, t.lag_close_21d,
    t.log_ret_1d, t.log_ret_5d, t.log_ret_21d,
    t.sma_5, t.sma_10, t.sma_21, t.sma_50, t.sma_200,
    t.volatility_21d,
    t.rsi_14,
    t.macd, t.macd_signal, t.macd_histogram,
    t.bb_upper, t.bb_middle, t.bb_lower, t.bb_position, t.bb_width_pct,
    t.atr_14, t.tr_pct,
    t.stoch_k, t.stoch_d,
    t.roc_10d, t.roc_21d, t.roc_63d, t.momentum_10d, t.momentum_21d,
    t.volume, t.avg_volume_21d, t.volume_ratio, t.volume_zscore, t.obv,
    
    -- ========================================================================
    -- CROSS-ASSET CORRELATIONS (11 features)
    -- ========================================================================
    c.corr_zl_zs_60d,
    c.corr_zl_zm_60d,
    c.corr_zl_cl_60d,
    c.corr_zl_ho_60d,
    c.corr_zl_hg_60d,
    c.corr_zl_dx_60d,
    c.corr_cl_ho_60d,
    c.corr_cl_rb_60d,
    c.corr_cl_dx_60d,
    c.corr_hg_gc_60d,
    c.corr_hg_dx_60d,
    
    -- ========================================================================
    -- FUNDAMENTAL SPREADS (6 features)
    -- ========================================================================
    s.board_crush_spread,
    s.oil_share_of_crush,
    s.boho_spread,
    s.crack_spread,
    s.china_copper_proxy,
    s.dollar_index,
    
    -- ========================================================================
    -- BIG 8 BUCKET SCORES (16 features: 8 scores + 8 key metrics)
    -- ========================================================================
    b.crush_bucket_score,
    b.china_bucket_score,
    b.fx_bucket_score,
    b.fed_bucket_score,
    b.tariff_bucket_score,
    b.biofuel_bucket_score,
    b.energy_bucket_score,
    b.volatility_bucket_score,
    b.board_crush,
    b.china_pulse,
    b.yield_curve_slope,
    b.tariff_activity,
    b.rin_d4,
    b.crude_price,
    b.vix,
    
    -- ========================================================================
    -- TARGETS (8 features: 4 price + 4 return)
    -- ========================================================================
    tgt.target_price_1w,
    tgt.target_price_1m,
    tgt.target_price_3m,
    tgt.target_price_6m,
    tgt.target_ret_1w,
    tgt.target_ret_1m,
    tgt.target_ret_3m,
    tgt.target_ret_6m

FROM tech_indicators t
LEFT JOIN targets tgt USING (as_of_date, symbol)
LEFT JOIN correlations c USING (as_of_date)
LEFT JOIN spreads s USING (as_of_date)
LEFT JOIN buckets b USING (as_of_date)
ORDER BY as_of_date;

-- ============================================================================
-- MACRO: Build Feature Matrix for ALL Symbols (31 Total)
-- ============================================================================
CREATE OR REPLACE MACRO build_all_symbols_features() AS TABLE
-- Agricultural (8)
SELECT * FROM build_symbol_features('ZL')
UNION ALL SELECT * FROM build_symbol_features('ZS')
UNION ALL SELECT * FROM build_symbol_features('ZM')
UNION ALL SELECT * FROM build_symbol_features('ZC')
UNION ALL SELECT * FROM build_symbol_features('ZW')
UNION ALL SELECT * FROM build_symbol_features('ZO')
UNION ALL SELECT * FROM build_symbol_features('HE')
UNION ALL SELECT * FROM build_symbol_features('FCPO')
-- Energy (4)
UNION ALL SELECT * FROM build_symbol_features('CL')
UNION ALL SELECT * FROM build_symbol_features('HO')
UNION ALL SELECT * FROM build_symbol_features('RB')
UNION ALL SELECT * FROM build_symbol_features('NG')
-- Metals (5)
UNION ALL SELECT * FROM build_symbol_features('HG')
UNION ALL SELECT * FROM build_symbol_features('GC')
UNION ALL SELECT * FROM build_symbol_features('SI')
UNION ALL SELECT * FROM build_symbol_features('PL')
UNION ALL SELECT * FROM build_symbol_features('PA')
-- Treasuries (3)
UNION ALL SELECT * FROM build_symbol_features('ZF')
UNION ALL SELECT * FROM build_symbol_features('ZN')
UNION ALL SELECT * FROM build_symbol_features('ZB')
-- FX Futures (10)
UNION ALL SELECT * FROM build_symbol_features('6E')
UNION ALL SELECT * FROM build_symbol_features('6J')
UNION ALL SELECT * FROM build_symbol_features('6B')
UNION ALL SELECT * FROM build_symbol_features('6C')
UNION ALL SELECT * FROM build_symbol_features('6A')
UNION ALL SELECT * FROM build_symbol_features('6N')
UNION ALL SELECT * FROM build_symbol_features('6M')
UNION ALL SELECT * FROM build_symbol_features('6L')
UNION ALL SELECT * FROM build_symbol_features('6S')
UNION ALL SELECT * FROM build_symbol_features('DX');

