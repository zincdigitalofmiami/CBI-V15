-- ============================================================================
-- Trump Social Media Market Impact Analysis View
-- ============================================================================
-- Purpose: Correlate Trump posts with market reactions (VIX, ZL, tariff sentiment)
-- Use case: Calculate posting frequency vs volatility, price moves, sentiment shifts
-- ============================================================================

CREATE OR REPLACE VIEW raw.v_trump_market_impact AS
WITH trump_posts AS (
    SELECT 
        DATE_TRUNC('day', published_date) as post_date,
        COUNT(*) as posts_count,
        COUNT(CASE WHEN LOWER(content) LIKE '%china%' THEN 1 END) as china_mentions,
        COUNT(CASE WHEN LOWER(content) LIKE '%tariff%' THEN 1 END) as tariff_mentions,
        COUNT(CASE WHEN LOWER(content) LIKE '%trade%' THEN 1 END) as trade_mentions,
        COUNT(CASE WHEN LOWER(content) LIKE '%soybean%' OR LOWER(content) LIKE '%farmer%' THEN 1 END) as ag_mentions,
        AVG(sentiment_score) as avg_sentiment,
        AVG(zl_impact_score) as avg_zl_impact
    FROM raw.scrapecreators_trump
    GROUP BY DATE_TRUNC('day', published_date)
),
market_data AS (
    SELECT 
        as_of_date,
        close as zl_close,
        LAG(close, 1) OVER (ORDER BY as_of_date) as zl_prev_close,
        (close - LAG(close, 1) OVER (ORDER BY as_of_date)) / LAG(close, 1) OVER (ORDER BY as_of_date) * 100 as zl_pct_change
    FROM raw.databento_futures_ohlcv_1d
    WHERE symbol = 'ZL'
),
vix_data AS (
    SELECT 
        date as vix_date,
        value as vix_close,
        LAG(value, 1) OVER (ORDER BY date) as vix_prev_close,
        (value - LAG(value, 1) OVER (ORDER BY date)) as vix_change
    FROM raw.fred_economic
    WHERE series_id = 'VIXCLS'
)
SELECT 
    tp.post_date,
    tp.posts_count,
    tp.china_mentions,
    tp.tariff_mentions,
    tp.trade_mentions,
    tp.ag_mentions,
    tp.avg_sentiment,
    tp.avg_zl_impact,
    
    -- Market reactions (same day)
    md.zl_close,
    md.zl_pct_change,
    vd.vix_close,
    vd.vix_change,
    
    -- Next day reactions (T+1)
    LEAD(md.zl_pct_change, 1) OVER (ORDER BY tp.post_date) as zl_pct_change_t1,
    LEAD(vd.vix_change, 1) OVER (ORDER BY tp.post_date) as vix_change_t1,
    
    -- Classify posting intensity
    CASE 
        WHEN tp.posts_count >= 10 THEN 'HIGH'
        WHEN tp.posts_count >= 5 THEN 'MEDIUM'
        ELSE 'LOW'
    END as posting_intensity,
    
    -- Classify topic focus
    CASE 
        WHEN tp.china_mentions > 0 AND tp.tariff_mentions > 0 THEN 'CHINA_TARIFFS'
        WHEN tp.china_mentions > 0 THEN 'CHINA'
        WHEN tp.tariff_mentions > 0 THEN 'TARIFFS'
        WHEN tp.trade_mentions > 0 THEN 'TRADE'
        WHEN tp.ag_mentions > 0 THEN 'AGRICULTURE'
        ELSE 'OTHER'
    END as topic_focus

FROM trump_posts tp
LEFT JOIN market_data md ON tp.post_date = md.as_of_date
LEFT JOIN vix_data vd ON tp.post_date = vd.vix_date
ORDER BY tp.post_date DESC;

-- ============================================================================
-- Usage Examples:
-- ============================================================================

-- 1. Correlation between Trump posting frequency and VIX spikes
-- SELECT 
--     posting_intensity,
--     AVG(vix_change) as avg_vix_change,
--     AVG(vix_change_t1) as avg_vix_change_next_day
-- FROM raw.v_trump_market_impact
-- WHERE post_date >= '2024-01-01'
-- GROUP BY posting_intensity;

-- 2. ZL price reaction to China/tariff posts
-- SELECT 
--     topic_focus,
--     COUNT(*) as days,
--     AVG(zl_pct_change) as avg_same_day_move,
--     AVG(zl_pct_change_t1) as avg_next_day_move
-- FROM raw.v_trump_market_impact
-- WHERE topic_focus IN ('CHINA_TARIFFS', 'CHINA', 'TARIFFS')
-- GROUP BY topic_focus;

-- 3. High-frequency posting days and market volatility
-- SELECT 
--     post_date,
--     posts_count,
--     china_mentions + tariff_mentions as policy_mentions,
--     zl_pct_change,
--     vix_change
-- FROM raw.v_trump_market_impact
-- WHERE posts_count >= 10
-- ORDER BY post_date DESC;


