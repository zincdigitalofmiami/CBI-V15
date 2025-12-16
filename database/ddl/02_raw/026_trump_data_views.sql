-- ============================================================================
-- Trump Data Separation Views
-- ============================================================================
-- Purpose: Clearly separate Trump's direct posts from news ABOUT Trump
-- ============================================================================

-- View 1: Trump Direct Posts Only (from Twitter/Truth Social/Facebook)
CREATE OR REPLACE VIEW raw.v_trump_direct_posts AS
SELECT 
    post_id as article_id,
    published_date as date,
    platform,
    content as text,
    sentiment_score,
    zl_impact_score,
    'TRUMP_DIRECT' as source_type,
    source,
    ingested_at
FROM raw.scrapecreators_trump
WHERE platform IN ('twitter', 'truthsocial', 'facebook');

-- View 2: News ABOUT Trump (from Google/media)
CREATE OR REPLACE VIEW raw.v_trump_news_mentions AS
SELECT 
    article_id,
    date,
    headline,
    content as text,
    bucket_name,
    policy_axis,
    zl_sentiment,
    sentiment_score,
    'TRUMP_NEWS' as source_type,
    source,
    url,
    created_at as ingested_at
FROM raw.scrapecreators_news_buckets
WHERE is_trump_related = true
  AND source != 'twitter';  -- Exclude direct tweets (those go in v_trump_direct_posts)

-- View 3: Combined Trump Activity (for correlation analysis)
CREATE OR REPLACE VIEW raw.v_trump_all_activity AS
SELECT 
    article_id,
    date,
    text,
    source_type,
    source,
    COALESCE(sentiment_score, 
             CASE zl_sentiment 
                 WHEN 'bullish' THEN 0.7
                 WHEN 'bearish' THEN -0.7
                 ELSE 0.0
             END) as sentiment_score,
    zl_impact_score,
    ingested_at
FROM (
    -- Direct posts
    SELECT * FROM raw.v_trump_direct_posts
    
    UNION ALL
    
    -- News mentions
    SELECT 
        article_id,
        date,
        COALESCE(headline, '') || ' ' || COALESCE(text, '') as text,
        source_type,
        source,
        NULL as sentiment_score,
        NULL as zl_impact_score,
        ingested_at
    FROM raw.v_trump_news_mentions
) combined
ORDER BY date DESC, ingested_at DESC;

-- View 4: Daily Trump Activity Summary
CREATE OR REPLACE VIEW raw.v_trump_daily_summary AS
SELECT 
    date,
    COUNT(*) as total_items,
    COUNT(CASE WHEN source_type = 'TRUMP_DIRECT' THEN 1 END) as direct_posts,
    COUNT(CASE WHEN source_type = 'TRUMP_NEWS' THEN 1 END) as news_mentions,
    AVG(sentiment_score) as avg_sentiment,
    AVG(zl_impact_score) as avg_zl_impact,
    
    -- Topic counts (from text analysis)
    COUNT(CASE WHEN LOWER(text) LIKE '%china%' THEN 1 END) as china_mentions,
    COUNT(CASE WHEN LOWER(text) LIKE '%tariff%' THEN 1 END) as tariff_mentions,
    COUNT(CASE WHEN LOWER(text) LIKE '%trade%' THEN 1 END) as trade_mentions,
    COUNT(CASE WHEN LOWER(text) LIKE '%soybean%' OR LOWER(text) LIKE '%farmer%' THEN 1 END) as ag_mentions,
    
    -- Intensity classification
    CASE 
        WHEN COUNT(CASE WHEN source_type = 'TRUMP_DIRECT' THEN 1 END) >= 10 THEN 'HIGH'
        WHEN COUNT(CASE WHEN source_type = 'TRUMP_DIRECT' THEN 1 END) >= 5 THEN 'MEDIUM'
        WHEN COUNT(CASE WHEN source_type = 'TRUMP_DIRECT' THEN 1 END) > 0 THEN 'LOW'
        ELSE 'NONE'
    END as posting_intensity
    
FROM raw.v_trump_all_activity
GROUP BY date
ORDER BY date DESC;

-- ============================================================================
-- Usage Examples:
-- ============================================================================

-- 1. Get Trump's direct posts only
-- SELECT * FROM raw.v_trump_direct_posts ORDER BY date DESC LIMIT 10;

-- 2. Get news about Trump
-- SELECT * FROM raw.v_trump_news_mentions ORDER BY date DESC LIMIT 10;

-- 3. Daily summary with market correlation
-- SELECT 
--     t.date,
--     t.direct_posts,
--     t.news_mentions,
--     t.avg_sentiment,
--     t.china_mentions,
--     t.tariff_mentions,
--     d.close as zl_close,
--     (d.close - LAG(d.close) OVER (ORDER BY d.as_of_date)) / LAG(d.close) OVER (ORDER BY d.as_of_date) * 100 as zl_pct_change
-- FROM raw.v_trump_daily_summary t
-- LEFT JOIN raw.databento_futures_ohlcv_1d d ON t.date = d.as_of_date AND d.symbol = 'ZL'
-- WHERE t.date >= '2024-01-01'
-- ORDER BY t.date DESC;
