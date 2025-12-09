-- ============================================================================
-- CFTC Commitment of Traders (COT) Data Tables
-- ============================================================================
-- Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
-- Frequency: Weekly (released Friday 3:30 PM ET, data as of prior Tuesday)
-- Coverage: All futures symbols (38 total)
-- ============================================================================

-- ============================================================================
-- Table: raw.cftc_cot_disaggregated
-- ============================================================================
-- Disaggregated COT report for commodity futures
-- Trader categories: Producer/Merchant, Swap Dealers, Managed Money, Other Reportable
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.cftc_cot_disaggregated (
    report_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    open_interest BIGINT,
    
    -- Producer/Merchant/Processor/User positions
    prod_merc_long BIGINT,
    prod_merc_short BIGINT,
    prod_merc_net BIGINT,
    prod_merc_net_pct_oi DOUBLE,
    
    -- Swap Dealer positions
    swap_long BIGINT,
    swap_short BIGINT,
    swap_net BIGINT,
    
    -- Managed Money positions (hedge funds, CTAs)
    managed_money_long BIGINT,
    managed_money_short BIGINT,
    managed_money_net BIGINT,
    managed_money_net_pct_oi DOUBLE,
    
    -- Other Reportable positions
    other_rept_long BIGINT,
    other_rept_short BIGINT,
    other_rept_net BIGINT,
    
    -- Non-Reportable positions (small traders)
    nonrept_long BIGINT,
    nonrept_short BIGINT,
    nonrept_net BIGINT,
    
    PRIMARY KEY (report_date, symbol)
);

COMMENT ON TABLE raw.cftc_cot_disaggregated IS 'CFTC Disaggregated COT report for commodity futures';

-- ============================================================================
-- Table: raw.cftc_cot_tff
-- ============================================================================
-- Traders in Financial Futures (TFF) report for FX and Treasury futures
-- Trader categories: Dealer/Intermediary, Asset Manager, Leveraged Funds, Other Reportable
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw.cftc_cot_tff (
    report_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    open_interest BIGINT,
    
    -- Dealer/Intermediary positions
    dealer_long BIGINT,
    dealer_short BIGINT,
    dealer_net BIGINT,
    
    -- Asset Manager/Institutional positions
    asset_mgr_long BIGINT,
    asset_mgr_short BIGINT,
    asset_mgr_net BIGINT,
    asset_mgr_net_pct_oi DOUBLE,
    
    -- Leveraged Funds positions (hedge funds, CTAs)
    lev_money_long BIGINT,
    lev_money_short BIGINT,
    lev_money_net BIGINT,
    lev_money_net_pct_oi DOUBLE,
    
    -- Other Reportable positions
    other_rept_long BIGINT,
    other_rept_short BIGINT,
    other_rept_net BIGINT,
    
    -- Non-Reportable positions (small traders)
    nonrept_long BIGINT,
    nonrept_short BIGINT,
    nonrept_net BIGINT,
    
    PRIMARY KEY (report_date, symbol)
);

COMMENT ON TABLE raw.cftc_cot_tff IS 'CFTC Traders in Financial Futures (TFF) report for FX and Treasury futures';

-- ============================================================================
-- Indexes for performance
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_cftc_cot_disagg_symbol_date 
ON raw.cftc_cot_disaggregated(symbol, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_cftc_cot_tff_symbol_date 
ON raw.cftc_cot_tff(symbol, report_date DESC);

-- ============================================================================
-- Example Queries
-- ============================================================================

-- Get latest COT data for ZL (Soybean Oil)
-- SELECT * FROM raw.cftc_cot_disaggregated 
-- WHERE symbol = 'ZL' 
-- ORDER BY report_date DESC 
-- LIMIT 10;

-- Get managed money net positions for all commodities
-- SELECT 
--     symbol,
--     report_date,
--     managed_money_net,
--     managed_money_net_pct_oi
-- FROM raw.cftc_cot_disaggregated
-- WHERE report_date = (SELECT MAX(report_date) FROM raw.cftc_cot_disaggregated)
-- ORDER BY managed_money_net_pct_oi DESC;

-- Get leveraged funds positions for FX futures
-- SELECT 
--     symbol,
--     report_date,
--     lev_money_net,
--     lev_money_net_pct_oi
-- FROM raw.cftc_cot_tff
-- WHERE report_date = (SELECT MAX(report_date) FROM raw.cftc_cot_tff)
-- ORDER BY lev_money_net_pct_oi DESC;

-- Calculate week-over-week change in managed money positions
-- WITH current_week AS (
--     SELECT * FROM raw.cftc_cot_disaggregated
--     WHERE report_date = (SELECT MAX(report_date) FROM raw.cftc_cot_disaggregated)
-- ),
-- prior_week AS (
--     SELECT * FROM raw.cftc_cot_disaggregated
--     WHERE report_date = (SELECT MAX(report_date) FROM raw.cftc_cot_disaggregated WHERE report_date < (SELECT MAX(report_date) FROM raw.cftc_cot_disaggregated))
-- )
-- SELECT 
--     c.symbol,
--     c.report_date,
--     c.managed_money_net AS current_net,
--     p.managed_money_net AS prior_net,
--     c.managed_money_net - p.managed_money_net AS net_change,
--     ((c.managed_money_net - p.managed_money_net) / NULLIF(p.managed_money_net, 0) * 100) AS pct_change
-- FROM current_week c
-- JOIN prior_week p ON c.symbol = p.symbol
-- ORDER BY ABS(net_change) DESC;

