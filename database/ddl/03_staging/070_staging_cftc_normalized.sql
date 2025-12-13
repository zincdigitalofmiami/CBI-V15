-- Staging: CFTC COT Normalized
-- Cleaned and derived CFTC positioning metrics

CREATE TABLE IF NOT EXISTS staging.cftc_normalized (
    date DATE NOT NULL,
    commodity VARCHAR NOT NULL,
    -- Net positions
    managed_money_net BIGINT,
    producer_merchant_net BIGINT,
    swap_dealer_net BIGINT,
    -- Percentile ranks (0-100)
    managed_money_net_pctile DECIMAL(5, 2),
    producer_merchant_net_pctile DECIMAL(5, 2),
    -- Changes
    managed_money_net_change_1w BIGINT,
    managed_money_net_change_4w BIGINT,
    -- Concentration
    total_open_interest BIGINT,
    managed_money_pct_of_oi DECIMAL(5, 4),
    -- Extremes
    is_managed_money_extreme_long BOOLEAN,  -- > 90th percentile
    is_managed_money_extreme_short BOOLEAN, -- < 10th percentile
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, commodity)
);

-- Weekly COT data forward-filled to daily grain
-- Percentiles computed over 52-week lookback

CREATE INDEX IF NOT EXISTS idx_cftc_normalized_commodity 
    ON staging.cftc_normalized(commodity);

