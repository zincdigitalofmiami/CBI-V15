-- Raw CFTC Commitments of Traders
-- Weekly COT reports (disaggregated futures)
-- Collector transforms CFTC data: maps contracts to symbols, computes net and pct_oi
--
-- Data source: https://www.cftc.gov/MarketReports/CommitmentsofTraders
-- Reports released: Every Friday at 3:30 PM ET (data as of prior Tuesday)

CREATE TABLE IF NOT EXISTS raw.cftc_cot (
    report_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,  -- Our symbol mapping (ZL, ZS, ZM, etc.)
    -- Positions (vendor columns, shortened names)
    open_interest BIGINT,
    prod_merc_long BIGINT,
    prod_merc_short BIGINT,
    swap_long BIGINT,
    swap_short BIGINT,
    managed_money_long BIGINT,
    managed_money_short BIGINT,
    other_rept_long BIGINT,
    other_rept_short BIGINT,
    nonrept_long BIGINT,
    nonrept_short BIGINT,
    -- Computed by collector (net = long - short)
    prod_merc_net BIGINT,
    swap_net BIGINT,
    managed_money_net BIGINT,
    other_rept_net BIGINT,
    nonrept_net BIGINT,
    -- Computed: net as % of open interest
    managed_money_net_pct_oi DECIMAL(8, 4),
    prod_merc_net_pct_oi DECIMAL(8, 4),
    source VARCHAR DEFAULT 'cftc_disagg',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, symbol)
);

-- Traders in Financial Futures (TFF) table
-- Different report format with dealer/asset manager/leveraged money categories
CREATE TABLE IF NOT EXISTS raw.cftc_cot_tff (
    report_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    open_interest BIGINT,
    dealer_long BIGINT,
    dealer_short BIGINT,
    asset_mgr_long BIGINT,
    asset_mgr_short BIGINT,
    lev_money_long BIGINT,
    lev_money_short BIGINT,
    other_rept_long BIGINT,
    other_rept_short BIGINT,
    nonrept_long BIGINT,
    nonrept_short BIGINT,
    -- Computed by collector
    dealer_net BIGINT,
    asset_mgr_net BIGINT,
    lev_money_net BIGINT,
    other_rept_net BIGINT,
    nonrept_net BIGINT,
    -- Computed: net as % of open interest
    lev_money_net_pct_oi DECIMAL(8, 4),
    asset_mgr_net_pct_oi DECIMAL(8, 4),
    source VARCHAR DEFAULT 'cftc_tff',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, symbol)
);

-- Key symbols: ZL, ZS, ZM, ZC, ZW, CL, HO, GC, SI, HG, DX, 6E, 6J, etc.

CREATE INDEX IF NOT EXISTS idx_cftc_cot_symbol 
    ON raw.cftc_cot(symbol);

CREATE INDEX IF NOT EXISTS idx_cftc_cot_tff_symbol 
    ON raw.cftc_cot_tff(symbol);
