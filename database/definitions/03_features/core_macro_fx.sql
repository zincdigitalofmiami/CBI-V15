-- ============================================================================
-- CORE MACRO/FX FEATURE VIEW
-- ============================================================================
-- Base features inherited by ALL Big 8 bucket specialists + main ZL predictor.
-- ~50 features: FX (16) + Macro (20) + Price/Volume (3) + Cross-Asset (5)
--
-- ALL buckets get these features as their foundation,
-- then ADD bucket-specific features on top (ADDITIVE model).
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS features;

CREATE OR REPLACE VIEW features.core_macro_fx AS
WITH 
-- Get all ZL trading days (primary calendar)
zl_calendar AS (
    SELECT DISTINCT as_of_date
    FROM raw.databento_ohlcv_daily
    WHERE symbol = 'ZL'
),

-- Databento prices (pivot to wide format)
futures_prices AS (
    SELECT
        as_of_date,
        MAX(CASE WHEN symbol = 'ZL' THEN close END) AS zl_close,
        MAX(CASE WHEN symbol = 'ZL' THEN volume END) AS zl_volume,
        MAX(CASE WHEN symbol = 'ZL' THEN open_interest END) AS zl_open_interest,
        MAX(CASE WHEN symbol = 'ZS' THEN close END) AS zs_close,
        MAX(CASE WHEN symbol = 'ZM' THEN close END) AS zm_close,
        MAX(CASE WHEN symbol = 'CL' THEN close END) AS cl_close,
        MAX(CASE WHEN symbol = 'HO' THEN close END) AS ho_close,
        MAX(CASE WHEN symbol = 'HG' THEN close END) AS hg_close,
        MAX(CASE WHEN symbol = '6L' THEN close END) AS brl_close,  -- BRL futures
        MAX(CASE WHEN symbol = 'DX' THEN close END) AS dxy_close   -- Dollar Index
    FROM raw.databento_ohlcv_daily
    WHERE symbol IN ('ZL', 'ZS', 'ZM', 'CL', 'HO', 'HG', '6L', 'DX')
    GROUP BY as_of_date
),

-- FRED macro data (pivot to wide format)
fred_macro AS (
    SELECT
        date AS as_of_date,
        MAX(CASE WHEN series_id = 'FEDFUNDS' THEN value END) AS fedfunds,
        MAX(CASE WHEN series_id = 'DFEDTARU' THEN value END) AS dfedtaru,
        MAX(CASE WHEN series_id = 'DGS10' THEN value END) AS dgs10,
        MAX(CASE WHEN series_id = 'DGS2' THEN value END) AS dgs2,
        MAX(CASE WHEN series_id = 'DGS3MO' THEN value END) AS dgs3mo,
        MAX(CASE WHEN series_id = 'T10Y2Y' THEN value END) AS t10y2y,
        MAX(CASE WHEN series_id = 'T10Y3M' THEN value END) AS t10y3m,
        MAX(CASE WHEN series_id = 'NFCI' THEN value END) AS nfci,
        MAX(CASE WHEN series_id = 'STLFSI4' THEN value END) AS stlfsi4,
        MAX(CASE WHEN series_id = 'VIXCLS' THEN value END) AS vixcls,
        MAX(CASE WHEN series_id = 'UNRATE' THEN value END) AS unrate,
        MAX(CASE WHEN series_id = 'CPIAUCSL' THEN value END) AS cpiaucsl
    FROM raw.fred_observations
    WHERE series_id IN ('FEDFUNDS', 'DFEDTARU', 'DGS10', 'DGS2', 'DGS3MO', 
                        'T10Y2Y', 'T10Y3M', 'NFCI', 'STLFSI4', 'VIXCLS', 
                        'UNRATE', 'CPIAUCSL')
    GROUP BY date
),

-- FX Momentum (multi-horizon)
fx_momentum AS (
    SELECT
        as_of_date,
        -- BRL momentum (21d, 63d, 252d)
        (brl_close - LAG(brl_close, 21) OVER (ORDER BY as_of_date)) / NULLIF(LAG(brl_close, 21) OVER (ORDER BY as_of_date), 0) AS brl_momentum_21d,
        (brl_close - LAG(brl_close, 63) OVER (ORDER BY as_of_date)) / NULLIF(LAG(brl_close, 63) OVER (ORDER BY as_of_date), 0) AS brl_momentum_63d,
        (brl_close - LAG(brl_close, 252) OVER (ORDER BY as_of_date)) / NULLIF(LAG(brl_close, 252) OVER (ORDER BY as_of_date), 0) AS brl_momentum_252d,
        -- DXY momentum (21d, 63d, 252d)
        (dxy_close - LAG(dxy_close, 21) OVER (ORDER BY as_of_date)) / NULLIF(LAG(dxy_close, 21) OVER (ORDER BY as_of_date), 0) AS dxy_momentum_21d,
        (dxy_close - LAG(dxy_close, 63) OVER (ORDER BY as_of_date)) / NULLIF(LAG(dxy_close, 63) OVER (ORDER BY as_of_date), 0) AS dxy_momentum_63d,
        (dxy_close - LAG(dxy_close, 252) OVER (ORDER BY as_of_date)) / NULLIF(LAG(dxy_close, 252) OVER (ORDER BY as_of_date), 0) AS dxy_momentum_252d,
        -- BRL volatility (21d, 63d realized vol)
        STDDEV(LN(brl_close / LAG(brl_close, 1) OVER (ORDER BY as_of_date))) OVER (ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) * SQRT(252) AS brl_volatility_21d,
        STDDEV(LN(brl_close / LAG(brl_close, 1) OVER (ORDER BY as_of_date))) OVER (ORDER BY as_of_date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW) * SQRT(252) AS brl_volatility_63d
    FROM futures_prices
),

-- ZL-BRL and ZL-DXY Correlations (multi-horizon)
fx_correlations AS (
    SELECT
        as_of_date,
        -- ZL-BRL correlation (30d, 60d, 90d)
        CORR(
            LN(zl_close / LAG(zl_close, 1) OVER (ORDER BY as_of_date)),
            LN(brl_close / LAG(brl_close, 1) OVER (ORDER BY as_of_date))
        ) OVER (ORDER BY as_of_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS corr_zl_brl_30d,
        CORR(
            LN(zl_close / LAG(zl_close, 1) OVER (ORDER BY as_of_date)),
            LN(brl_close / LAG(brl_close, 1) OVER (ORDER BY as_of_date))
        ) OVER (ORDER BY as_of_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS corr_zl_brl_60d,
        CORR(
            LN(zl_close / LAG(zl_close, 1) OVER (ORDER BY as_of_date)),
            LN(brl_close / LAG(brl_close, 1) OVER (ORDER BY as_of_date))
        ) OVER (ORDER BY as_of_date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW) AS corr_zl_brl_90d,
        -- ZL-DXY correlation (30d, 60d, 90d)
        CORR(
            LN(zl_close / LAG(zl_close, 1) OVER (ORDER BY as_of_date)),
            LN(dxy_close / LAG(dxy_close, 1) OVER (ORDER BY as_of_date))
        ) OVER (ORDER BY as_of_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS corr_zl_dxy_30d,
        CORR(
            LN(zl_close / LAG(zl_close, 1) OVER (ORDER BY as_of_date)),
            LN(dxy_close / LAG(dxy_close, 1) OVER (ORDER BY as_of_date))
        ) OVER (ORDER BY as_of_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS corr_zl_dxy_60d,
        CORR(
            LN(zl_close / LAG(zl_close, 1) OVER (ORDER BY as_of_date)),
            LN(dxy_close / LAG(dxy_close, 1) OVER (ORDER BY as_of_date))
        ) OVER (ORDER BY as_of_date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW) AS corr_zl_dxy_90d
    FROM futures_prices
)

-- FINAL SELECT: Join all components
SELECT
    c.as_of_date,
    
    -- ========================================================================
    -- FX FEATURES (16 features - 7 must-haves from FX review)
    -- ========================================================================
    
    -- 1. Rate Differential (Carry Trade)
    -- Note: Using Fed funds as proxy for US rate, would need BRL rate from FRED
    NULL AS rate_differential_brl_us,  -- TODO: Add when BRL rate available
    
    -- 2-4. BRL Momentum (3 features)
    fxm.brl_momentum_21d,
    fxm.brl_momentum_63d,
    fxm.brl_momentum_252d,
    
    -- 5-7. DXY Momentum (3 features)
    fxm.dxy_momentum_21d,
    fxm.dxy_momentum_63d,
    fxm.dxy_momentum_252d,
    
    -- 8-9. BRL Volatility (2 features)
    fxm.brl_volatility_21d,
    fxm.brl_volatility_63d,
    
    -- 10-12. ZL-BRL Correlation (3 features)
    fxc.corr_zl_brl_30d,
    fxc.corr_zl_brl_60d,
    fxc.corr_zl_brl_90d,
    
    -- 13-15. ZL-DXY Correlation (3 features)
    fxc.corr_zl_dxy_30d,
    fxc.corr_zl_dxy_60d,
    fxc.corr_zl_dxy_90d,
    
    -- 16. Terms of Trade (CRITICAL: Guard against zeros!)
    CASE 
        WHEN p.brl_close > 0 
        THEN p.zl_close / p.brl_close
        ELSE NULL
    END AS terms_of_trade,
    
    -- ========================================================================
    -- MACRO FEATURES (12 features from FRED)
    -- ========================================================================
    
    f.fedfunds AS fred_fedfunds,
    f.dfedtaru AS fred_dfedtaru,
    f.dgs10 AS fred_dgs10,
    f.dgs2 AS fred_dgs2,
    f.dgs3mo AS fred_dgs3mo,
    f.t10y2y AS fred_t10y2y,  -- Yield curve slope
    f.t10y3m AS fred_t10y3m,  -- Short-term curve
    f.nfci AS fred_nfci,      -- Financial conditions
    f.stlfsi4 AS fred_stlfsi4,  -- Financial stress
    f.vixcls AS fred_vixcls,  -- VIX volatility index
    f.unrate AS fred_unrate,  -- Unemployment rate
    f.cpiaucsl AS fred_cpiaucsl,  -- CPI inflation
    
    -- ========================================================================
    -- PRICE/VOLUME BASE (3 features)
    -- ========================================================================
    
    p.zl_close AS databento_zl_close,
    p.zl_volume AS databento_zl_volume,
    p.zl_open_interest AS databento_zl_open_interest,
    
    -- ========================================================================
    -- CROSS-ASSET (5 features - from cross_asset_features.sql)
    -- ========================================================================
    
    -- Board Crush: (ZM × 0.022 + ZL × 11) - ZS
    (p.zm_close * 0.022 + p.zl_close * 11) - p.zs_close AS board_crush_spread,
    
    -- Oil Share of Crush Value
    (p.zl_close * 11) / NULLIF((p.zm_close * 0.022 + p.zl_close * 11), 0) AS oil_share_of_crush,
    
    -- BOHO Spread: (ZL/100 × 7.5) - HO (Soy Oil vs Heating Oil)
    (p.zl_close / 100 * 7.5) - p.ho_close AS boho_spread,
    
    -- China Pulse: Copper as China demand proxy
    p.hg_close AS china_copper_proxy,
    
    -- Dollar Index
    p.dxy_close AS dollar_index

FROM zl_calendar c
LEFT JOIN futures_prices p USING (as_of_date)
LEFT JOIN fred_macro f USING (as_of_date)
LEFT JOIN fx_momentum fxm USING (as_of_date)
LEFT JOIN fx_correlations fxc USING (as_of_date)
ORDER BY c.as_of_date;

-- ============================================================================
-- FEATURE COUNT: ~50 features
-- ============================================================================
-- FX: 16 features (rate diff, BRL/DXY momentum, volatility, correlations, Terms of Trade)
-- Macro: 12 features (Fed funds, yields, curves, NFCI, STLFSI4, VIX, UNRATE, CPI)
-- Price/Volume: 3 features (ZL close, volume, OI)
-- Cross-Asset: 5 features (board crush, oil share, BOHO, HG proxy, DX)
-- ============================================================================



