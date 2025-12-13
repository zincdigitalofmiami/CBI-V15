-- Raw CFTC Commitments of Traders
-- Weekly COT reports

CREATE TABLE IF NOT EXISTS raw.cftc_cot (
    report_date DATE NOT NULL,
    commodity VARCHAR NOT NULL,
    -- Disaggregated report fields
    producer_merchant_long BIGINT,
    producer_merchant_short BIGINT,
    swap_dealer_long BIGINT,
    swap_dealer_short BIGINT,
    swap_dealer_spread BIGINT,
    managed_money_long BIGINT,
    managed_money_short BIGINT,
    managed_money_spread BIGINT,
    other_reportable_long BIGINT,
    other_reportable_short BIGINT,
    other_reportable_spread BIGINT,
    nonreportable_long BIGINT,
    nonreportable_short BIGINT,
    total_open_interest BIGINT,
    source VARCHAR DEFAULT 'cftc',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (report_date, commodity)
);

-- Key commodities: ZL, ZS, ZM, CL, DX

CREATE INDEX IF NOT EXISTS idx_cftc_cot_commodity 
    ON raw.cftc_cot(commodity);

