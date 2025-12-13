-- CFTC COT Compatibility Views
-- Provides standardized column names for macro consumption
-- Data lives in raw.cftc_cot (disagg) and raw.cftc_cot_tff (financial futures)

-- Disaggregated format view with macro-compatible column names
CREATE OR REPLACE VIEW raw.cftc_cot_disaggregated AS
SELECT
    report_date,
    symbol,
    open_interest,
    prod_merc_long,
    prod_merc_short,
    swap_long,
    swap_short,
    managed_money_long,
    managed_money_short,
    other_rept_long,
    other_rept_short,
    nonrept_long,
    nonrept_short,
    prod_merc_net,
    swap_net,
    managed_money_net,
    other_rept_net,
    nonrept_net,
    managed_money_net_pct_oi,
    prod_merc_net_pct_oi,
    source,
    ingested_at
FROM raw.cftc_cot;

-- TFF (Traders in Financial Futures) format view
CREATE OR REPLACE VIEW raw.v_cftc_cot_tff AS
SELECT
    report_date,
    symbol,
    open_interest,
    dealer_long,
    dealer_short,
    asset_mgr_long,
    asset_mgr_short,
    lev_money_long,
    lev_money_short,
    other_rept_long,
    other_rept_short,
    nonrept_long,
    nonrept_short,
    dealer_net,
    asset_mgr_net,
    lev_money_net,
    other_rept_net,
    nonrept_net,
    lev_money_net_pct_oi,
    asset_mgr_net_pct_oi,
    source,
    ingested_at
FROM raw.cftc_cot_tff;
