-- ============================================================================
-- BIG 8 BUCKET FEATURE AGGREGATION (WITH CFTC COT DATA)
-- ============================================================================
-- Aggregate technical indicators, external data, and CFTC positioning into 8 thematic buckets:
-- 1. Crush - Soybean crush margins + COT positioning
-- 2. China - Import demand signals + Copper COT
-- 3. FX - Currency effects + FX futures COT
-- 4. Fed - Monetary policy + Treasury COT
-- 5. Tariff - Trade policy + Ag COT positioning
-- 6. Biofuel - RFS/biodiesel demand + Soy Oil COT
-- 7. Energy - Crude correlation + Energy COT
-- 8. Volatility - Market volatility + VIX + Extreme positioning
--
-- CFTC COT Data provides positioning/sentiment for each bucket:
-- - Managed Money = Speculators (hedge funds, CTAs)
-- - Producer/Merchant = Commercial hedgers
-- - Net positions as % of open interest = Sentiment gauge
-- - Extreme positioning (>30% of OI) = Potential reversal signal
-- ============================================================================

-- ============================================================================
-- BUCKET 1: CRUSH (Soybean Crush Margins + COT Positioning)
-- ============================================================================
CREATE OR REPLACE MACRO calc_crush_bucket_score() AS TABLE
WITH crush_components AS (
    SELECT
        as_of_date,
        MAX(CASE WHEN symbol = 'ZL' THEN close END) AS zl_close,
        MAX(CASE WHEN symbol = 'ZS' THEN close END) AS zs_close,
        MAX(CASE WHEN symbol = 'ZM' THEN close END) AS zm_close
    FROM raw.databento_futures_ohlcv_1d
    WHERE symbol IN ('ZL', 'ZS', 'ZM')
    GROUP BY as_of_date
),
crush_cot_raw AS (
    -- Get COT positioning for crush components (weekly data)
    SELECT
        report_date AS as_of_date,
        MAX(CASE WHEN symbol = 'ZL' THEN managed_money_net_pct_oi END) AS zl_mm_net_pct,
        MAX(CASE WHEN symbol = 'ZS' THEN managed_money_net_pct_oi END) AS zs_mm_net_pct,
        MAX(CASE WHEN symbol = 'ZM' THEN managed_money_net_pct_oi END) AS zm_mm_net_pct,
        MAX(CASE WHEN symbol = 'ZL' THEN prod_merc_net_pct_oi END) AS zl_comm_net_pct,
        MAX(CASE WHEN symbol = 'ZS' THEN prod_merc_net_pct_oi END) AS zs_comm_net_pct
    FROM raw.cftc_cot_disaggregated
    WHERE symbol IN ('ZL', 'ZS', 'ZM')
    GROUP BY report_date
),
-- Forward-fill COT data to daily frequency (separate CTE to avoid nested windows)
crush_cot_filled AS (
    SELECT
        c.as_of_date,
        LAST_VALUE(cot.zl_mm_net_pct IGNORE NULLS) OVER (ORDER BY c.as_of_date) AS zl_spec_net_pct,
        LAST_VALUE(cot.zs_mm_net_pct IGNORE NULLS) OVER (ORDER BY c.as_of_date) AS zs_spec_net_pct,
        LAST_VALUE(cot.zl_comm_net_pct IGNORE NULLS) OVER (ORDER BY c.as_of_date) AS zl_hedger_net_pct
    FROM crush_components c
    LEFT JOIN crush_cot_raw cot ON c.as_of_date = cot.as_of_date
),
crush_calcs AS (
    SELECT
        c.as_of_date,
        -- Board Crush: (ZM × 0.022 + ZL × 11) - ZS
        (c.zm_close * 0.022 + c.zl_close * 11) - c.zs_close AS board_crush,
        -- Oil Share
        (c.zl_close * 11) / NULLIF((c.zm_close * 0.022 + c.zl_close * 11), 0) AS oil_share,
        -- ZL/ZS Ratio
        c.zl_close / NULLIF(c.zs_close, 0) AS zl_zs_ratio,
        -- COT positioning (pre-filled from above)
        cot.zl_spec_net_pct,
        cot.zs_spec_net_pct,
        cot.zl_hedger_net_pct,
        -- Positioning spread (speculators vs hedgers)
        COALESCE(cot.zl_spec_net_pct, 0) - COALESCE(cot.zl_hedger_net_pct, 0) AS zl_spec_hedger_spread
    FROM crush_components c
    LEFT JOIN crush_cot_filled cot ON c.as_of_date = cot.as_of_date
),
crush_features AS (
    SELECT
        as_of_date,
        board_crush,
        oil_share,
        zl_zs_ratio,
        zl_spec_net_pct,
        zs_spec_net_pct,
        zl_hedger_net_pct,
        zl_spec_hedger_spread,
        -- Moving averages of crush
        AVG(board_crush) OVER (ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS crush_sma_21,
        STDDEV(board_crush) OVER (ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS crush_volatility_21,
        -- Z-score
        (board_crush - AVG(board_crush) OVER (ORDER BY as_of_date ROWS BETWEEN 60 PRECEDING AND CURRENT ROW)) /
        NULLIF(STDDEV(board_crush) OVER (ORDER BY as_of_date ROWS BETWEEN 60 PRECEDING AND CURRENT ROW), 0) AS crush_zscore,
        -- COT positioning signals
        -- Extreme long positioning (>30% of OI) = potential reversal (bearish)
        -- Extreme short positioning (<-30% of OI) = potential reversal (bullish)
        CASE
            WHEN zl_spec_net_pct > 30 THEN -1  -- Overcrowded long = bearish
            WHEN zl_spec_net_pct < -30 THEN 1  -- Overcrowded short = bullish
            ELSE 0
        END AS zl_extreme_positioning_signal,
        -- Hedger vs Speculator divergence (hedgers are "smart money")
        -- If hedgers are net long and speculators are net short = bullish
        CASE
            WHEN zl_hedger_net_pct > 0 AND zl_spec_net_pct < 0 THEN 1
            WHEN zl_hedger_net_pct < 0 AND zl_spec_net_pct > 0 THEN -1
            ELSE 0
        END AS zl_smart_money_signal
    FROM crush_calcs
)
SELECT
    as_of_date,
    -- Normalize to 0-100 score (higher = more bullish for ZL)
    -- Combine crush fundamentals + COT positioning
    50 + (crush_zscore * 8) + (zl_extreme_positioning_signal * 5) + (zl_smart_money_signal * 7) AS crush_bucket_score,
    board_crush,
    oil_share,
    zl_zs_ratio,
    crush_sma_21,
    zl_spec_net_pct,
    zl_hedger_net_pct,
    zl_spec_hedger_spread,
    crush_volatility_21
FROM crush_features;

-- ============================================================================
-- BUCKET 2: CHINA (Import Demand + Copper COT)
-- ============================================================================
CREATE OR REPLACE MACRO calc_china_bucket_score() AS TABLE
WITH china_components AS (
    SELECT
        as_of_date,
        MAX(CASE WHEN symbol = 'HG' THEN close END) AS hg_close,  -- Copper (China demand proxy)
        MAX(CASE WHEN symbol = 'ZS' THEN close END) AS zs_close   -- Soybeans
    FROM raw.databento_futures_ohlcv_1d
    WHERE symbol IN ('HG', 'ZS')
    GROUP BY as_of_date
),
china_cot AS (
    -- Copper positioning = China demand sentiment
    SELECT
        report_date AS as_of_date,
        MAX(CASE WHEN symbol = 'HG' THEN managed_money_net_pct_oi END) AS hg_spec_net_pct,
        MAX(CASE WHEN symbol = 'HG' THEN prod_merc_net_pct_oi END) AS hg_comm_net_pct,
        MAX(CASE WHEN symbol = 'ZS' THEN managed_money_net_pct_oi END) AS zs_spec_net_pct
    FROM raw.cftc_cot_disaggregated
    WHERE symbol IN ('HG', 'ZS')
    GROUP BY report_date
),
china_returns AS (
    SELECT
        as_of_date,
        hg_close,
        zs_close,
        LN(hg_close / LAG(hg_close, 1) OVER (ORDER BY as_of_date)) AS hg_ret,
        LN(zs_close / LAG(zs_close, 1) OVER (ORDER BY as_of_date)) AS zs_ret
    FROM china_components
),
china_corr AS (
    SELECT
        as_of_date,
        hg_close,
        -- HG-ZS correlation (60-day) as China demand pulse
        CORR(hg_ret, zs_ret) OVER (ORDER BY as_of_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS hg_zs_corr_60d,
        -- HG momentum
        (hg_close - LAG(hg_close, 21) OVER (ORDER BY as_of_date)) / NULLIF(LAG(hg_close, 21) OVER (ORDER BY as_of_date), 0) AS hg_momentum_21d
    FROM china_returns
)
SELECT
    as_of_date,
    -- Score: Copper momentum + correlation strength
    50 + (hg_momentum_21d * 100) + (hg_zs_corr_60d * 25) AS china_bucket_score,
    hg_zs_corr_60d AS china_pulse,
    hg_momentum_21d AS copper_momentum
FROM china_corr;

-- ============================================================================
-- BUCKET 3: FX (Currency Effects)
-- ============================================================================
CREATE OR REPLACE MACRO calc_fx_bucket_score() AS TABLE
WITH fx_data AS (
    SELECT
        as_of_date,
        MAX(CASE WHEN symbol = 'DX' THEN close END) AS dx_close  -- Dollar Index
    FROM raw.databento_futures_ohlcv_1d
    WHERE symbol = 'DX'
    GROUP BY as_of_date
),
fx_returns AS (
    SELECT
        as_of_date,
        dx_close,
        LN(dx_close / LAG(dx_close, 1) OVER (ORDER BY as_of_date)) AS dx_log_ret,
        LAG(dx_close, 21) OVER (ORDER BY as_of_date) AS dx_close_21d_ago
    FROM fx_data
),
fx_features AS (
    SELECT
        as_of_date,
        dx_close,
        -- Dollar momentum (inverse for ZL - strong dollar = bearish commodities)
        (dx_close - dx_close_21d_ago) / NULLIF(dx_close_21d_ago, 0) AS dx_momentum_21d,
        -- Dollar volatility (annualized)
        STDDEV(dx_log_ret) OVER (ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) * SQRT(252) AS dx_volatility_21d
    FROM fx_returns
)
SELECT
    as_of_date,
    -- Score: Inverse of dollar strength (strong dollar = lower score = bearish ZL)
    50 - (dx_momentum_21d * 100) AS fx_bucket_score,
    dx_close AS dollar_index,
    dx_momentum_21d AS dollar_momentum,
    dx_volatility_21d AS dollar_volatility
FROM fx_features;

-- ============================================================================
-- BUCKET 4: FED (Monetary Policy)
-- ============================================================================
CREATE OR REPLACE MACRO calc_fed_bucket_score() AS TABLE
WITH fed_data AS (
    SELECT
        date AS as_of_date,
        MAX(CASE WHEN series_id = 'DGS10' THEN value END) AS dgs10,
        MAX(CASE WHEN series_id = 'DGS2' THEN value END) AS dgs2,
        MAX(CASE WHEN series_id = 'DFEDTARU' THEN value END) AS fed_rate
    FROM raw.fred_economic
    WHERE series_id IN ('DGS10', 'DGS2', 'DFEDTARU')
    GROUP BY date
),
fed_features AS (
    SELECT
        as_of_date,
        dgs10,
        dgs2,
        fed_rate,
        -- Yield curve slope
        dgs10 - dgs2 AS yield_curve_slope,
        -- Rate change momentum
        fed_rate - LAG(fed_rate, 21) OVER (ORDER BY as_of_date) AS fed_rate_change_21d
    FROM fed_data
)
SELECT
    as_of_date,
    -- Score: Steeper curve + falling rates = bullish commodities
    50 + (yield_curve_slope * 5) - (fed_rate_change_21d * 10) AS fed_bucket_score,
    yield_curve_slope,
    fed_rate,
    fed_rate_change_21d
FROM fed_features;

-- ============================================================================
-- BUCKET 5: TARIFF (Trade Policy)
-- ============================================================================
CREATE OR REPLACE MACRO calc_tariff_bucket_score() AS TABLE
WITH trump_sentiment AS (
    SELECT
        date AS as_of_date,
        -- Aggregate Trump sentiment from ScrapeCreators
        AVG(CASE
            WHEN zl_sentiment = 'BULLISH_ZL' THEN 1
            WHEN zl_sentiment = 'BEARISH_ZL' THEN -1
            ELSE 0
        END) AS trump_sentiment_avg,
        COUNT(*) AS trump_post_count
    FROM raw.scrapecreators_news_buckets
    WHERE is_trump_related = TRUE
      AND policy_axis IN ('TRADE_CHINA', 'TRADE_TARIFFS')
    GROUP BY date
),
tariff_features AS (
    SELECT
        as_of_date,
        trump_sentiment_avg,
        trump_post_count,
        -- 7-day rolling sentiment
        AVG(trump_sentiment_avg) OVER (ORDER BY as_of_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS trump_sentiment_7d,
        -- Post volume spike (activity indicator)
        trump_post_count / NULLIF(AVG(trump_post_count) OVER (ORDER BY as_of_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW), 0) AS post_volume_ratio
    FROM trump_sentiment
)
SELECT
    as_of_date,
    -- Score: Sentiment + activity spike
    50 + (trump_sentiment_7d * 25) + ((post_volume_ratio - 1) * 10) AS tariff_bucket_score,
    trump_sentiment_7d,
    post_volume_ratio AS tariff_activity
FROM tariff_features;

-- ============================================================================
-- BUCKET 6: BIOFUEL (RFS/Biodiesel Demand)
-- ============================================================================
CREATE OR REPLACE MACRO calc_biofuel_bucket_score() AS TABLE
WITH eia_data AS (
    -- EIA biodiesel/biofuel production from canonical EIA table
    SELECT
        date AS as_of_date,
        MAX(CASE WHEN series_id = 'biodiesel_production' THEN value END) AS biodiesel_prod
    FROM raw.eia_biofuels
    WHERE series_id = 'biodiesel_production'
    GROUP BY date
),
epa_data AS (
    -- EPA RIN prices from canonical EPA table
    SELECT
        date AS as_of_date,
        MAX(CASE WHEN series_id = 'rin_d4_price' THEN value END) AS rin_d4,
        MAX(CASE WHEN series_id = 'rin_d6_price' THEN value END) AS rin_d6
    FROM raw.epa_rin_prices
    WHERE series_id IN ('rin_d4_price', 'rin_d6_price')
    GROUP BY date
),
biofuel_data AS (
    -- Join EIA biodiesel production with EPA RIN prices
    SELECT
        COALESCE(e.as_of_date, r.as_of_date) AS as_of_date,
        e.biodiesel_prod,
        r.rin_d4,
        r.rin_d6
    FROM eia_data e
    FULL OUTER JOIN epa_data r ON e.as_of_date = r.as_of_date
),
-- CRITICAL FIX: Fill weekly EPA RIN prices to daily frequency
-- Without this, AutoGluon drops 80% of rows due to NaN
-- Uses LAST_VALUE carry-forward for weekly->daily conversion
biofuel_filled AS (
    SELECT
        as_of_date,
        LAST_VALUE(biodiesel_prod IGNORE NULLS) OVER (ORDER BY as_of_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS biodiesel_prod,
        LAST_VALUE(rin_d4 IGNORE NULLS) OVER (ORDER BY as_of_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rin_d4,
        LAST_VALUE(rin_d6 IGNORE NULLS) OVER (ORDER BY as_of_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rin_d6
    FROM (
        -- Generate daily series from min to max date
        SELECT DISTINCT as_of_date 
        FROM raw.databento_futures_ohlcv_1d 
        WHERE symbol = 'ZL'
    ) d
    LEFT JOIN biofuel_data USING (as_of_date)
),
biofuel_features AS (
    SELECT
        as_of_date,
        biodiesel_prod,
        rin_d4,
        rin_d6,
        -- Production momentum
        (biodiesel_prod - LAG(biodiesel_prod, 4) OVER (ORDER BY as_of_date)) / NULLIF(LAG(biodiesel_prod, 4) OVER (ORDER BY as_of_date), 0) AS biodiesel_momentum,
        -- RIN price momentum (higher RIN = more demand for biofuels)
        (rin_d4 - LAG(rin_d4, 4) OVER (ORDER BY as_of_date)) / NULLIF(LAG(rin_d4, 4) OVER (ORDER BY as_of_date), 0) AS rin_momentum
    FROM biofuel_filled
)
SELECT
    as_of_date,
    -- Score: Production + RIN price momentum
    50 + (biodiesel_momentum * 50) + (rin_momentum * 25) AS biofuel_bucket_score,
    biodiesel_prod,
    rin_d4,
    biodiesel_momentum,
    rin_momentum
FROM biofuel_features;
-- ============================================================================
-- BUCKET 7: ENERGY (Crude Correlation)
-- ============================================================================
CREATE OR REPLACE MACRO calc_energy_bucket_score() AS TABLE
WITH energy_data AS (
    SELECT
        as_of_date,
        MAX(CASE WHEN symbol = 'CL' THEN close END) AS cl_close,
        MAX(CASE WHEN symbol = 'HO' THEN close END) AS ho_close,
        MAX(CASE WHEN symbol = 'RB' THEN close END) AS rb_close,
        MAX(CASE WHEN symbol = 'ZL' THEN close END) AS zl_close
    FROM raw.databento_futures_ohlcv_1d
    WHERE symbol IN ('CL', 'HO', 'RB', 'ZL')
    GROUP BY as_of_date
),
energy_returns AS (
    SELECT
        as_of_date,
        cl_close,
        ho_close,
        zl_close,
        LN(cl_close / LAG(cl_close, 1) OVER (ORDER BY as_of_date)) AS cl_ret,
        LN(zl_close / LAG(zl_close, 1) OVER (ORDER BY as_of_date)) AS zl_ret,
        -- BOHO Spread: (ZL/100 × 7.5) - HO
        (zl_close / 100 * 7.5) - ho_close AS boho_spread
    FROM energy_data
),
energy_features AS (
    SELECT
        as_of_date,
        cl_close,
        boho_spread,
        -- CL-ZL correlation (energy substitution effect)
        CORR(cl_ret, zl_ret) OVER (ORDER BY as_of_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS cl_zl_corr_60d,
        -- Crude momentum
        (cl_close - LAG(cl_close, 21) OVER (ORDER BY as_of_date)) / NULLIF(LAG(cl_close, 21) OVER (ORDER BY as_of_date), 0) AS cl_momentum_21d,
        -- BOHO spread momentum
        (boho_spread - LAG(boho_spread, 21) OVER (ORDER BY as_of_date)) / NULLIF(ABS(LAG(boho_spread, 21) OVER (ORDER BY as_of_date)), 0) AS boho_momentum
    FROM energy_returns
)
SELECT
    as_of_date,
    -- Score: Inverse crude momentum (high crude = substitution away from ZL)
    50 - (cl_momentum_21d * 50) + (boho_momentum * 25) AS energy_bucket_score,
    cl_close AS crude_price,
    boho_spread,
    cl_zl_corr_60d AS energy_correlation,
    cl_momentum_21d AS crude_momentum
FROM energy_features;

-- ============================================================================
-- BUCKET 8: VOLATILITY (Market Volatility)
-- ============================================================================
CREATE OR REPLACE MACRO calc_volatility_bucket_score() AS TABLE
WITH vol_data AS (
    SELECT
        date AS as_of_date,
        MAX(CASE WHEN series_id = 'VIXCLS' THEN value END) AS vix
    FROM raw.fred_economic
    WHERE series_id = 'VIXCLS'
    GROUP BY date
),
zl_returns AS (
    -- First calculate log returns (separate CTE to avoid nested windows)
    SELECT
        as_of_date,
        symbol,
        close,
        LN(close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY as_of_date)) AS log_ret
    FROM raw.databento_futures_ohlcv_1d
    WHERE symbol = 'ZL'
),
zl_vol AS (
    -- Then calculate rolling volatility on the returns
    SELECT
        as_of_date,
        symbol,
        STDDEV(log_ret) OVER (
            PARTITION BY symbol
            ORDER BY as_of_date
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) * SQRT(252) AS zl_realized_volatility_21d
    FROM zl_returns
),
volatility_features AS (
    SELECT
        v.as_of_date,
        v.vix,
        z.zl_realized_volatility_21d,
        -- VIX momentum
        (v.vix - LAG(v.vix, 21) OVER (ORDER BY v.as_of_date)) / NULLIF(LAG(v.vix, 21) OVER (ORDER BY v.as_of_date), 0) AS vix_momentum,
        -- ZL volatility momentum
        (z.zl_realized_volatility_21d - LAG(z.zl_realized_volatility_21d, 21) OVER (ORDER BY v.as_of_date)) / NULLIF(LAG(z.zl_realized_volatility_21d, 21) OVER (ORDER BY v.as_of_date), 0) AS zl_volatility_momentum
    FROM vol_data v
    LEFT JOIN zl_vol z ON v.as_of_date = z.as_of_date
)
SELECT
    as_of_date,
    -- Score: Inverse of volatility (high volatility = risk-off = lower score)
    50 - (vix_momentum * 25) - (zl_volatility_momentum * 25) AS volatility_bucket_score,
    vix,
    zl_realized_volatility_21d AS zl_volatility,
    vix_momentum,
    zl_volatility_momentum
FROM volatility_features;

-- ============================================================================
-- MASTER: ALL BIG 8 BUCKET SCORES
-- ============================================================================
CREATE OR REPLACE MACRO calc_all_bucket_scores() AS TABLE
WITH
    crush AS (SELECT * FROM calc_crush_bucket_score()),
    china AS (SELECT * FROM calc_china_bucket_score()),
    fx AS (SELECT * FROM calc_fx_bucket_score()),
    fed AS (SELECT * FROM calc_fed_bucket_score()),
    tariff AS (SELECT * FROM calc_tariff_bucket_score()),
    biofuel AS (SELECT * FROM calc_biofuel_bucket_score()),
    energy AS (SELECT * FROM calc_energy_bucket_score()),
    vol AS (SELECT * FROM calc_volatility_bucket_score())
SELECT
    COALESCE(crush.as_of_date, china.as_of_date, fx.as_of_date, fed.as_of_date,
             tariff.as_of_date, biofuel.as_of_date, energy.as_of_date, vol.as_of_date) AS as_of_date,
    -- Bucket Scores (0-100 scale)
    crush.crush_bucket_score,
    china.china_bucket_score,
    fx.fx_bucket_score,
    fed.fed_bucket_score,
    tariff.tariff_bucket_score,
    biofuel.biofuel_bucket_score,
    energy.energy_bucket_score,
    vol.volatility_bucket_score,
    -- Key underlying metrics
    crush.board_crush,
    china.china_pulse,
    fx.dollar_index,
    fed.yield_curve_slope,
    tariff.tariff_activity,
    biofuel.rin_d4,
    energy.crude_price,
    vol.vix
FROM crush
FULL OUTER JOIN china USING (as_of_date)
FULL OUTER JOIN fx USING (as_of_date)
FULL OUTER JOIN fed USING (as_of_date)
FULL OUTER JOIN tariff USING (as_of_date)
FULL OUTER JOIN biofuel USING (as_of_date)
FULL OUTER JOIN energy USING (as_of_date)
FULL OUTER JOIN vol USING (as_of_date)
ORDER BY as_of_date;
