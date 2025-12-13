-- Features: Targets
-- Forward return targets for each horizon

CREATE TABLE IF NOT EXISTS features.targets (
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL DEFAULT 'ZL',
    -- Forward returns (log returns)
    target_1w DECIMAL(10, 6),   -- 5 trading days
    target_1m DECIMAL(10, 6),   -- 21 trading days
    target_3m DECIMAL(10, 6),   -- 63 trading days
    target_6m DECIMAL(10, 6),   -- 126 trading days
    -- Direction labels
    direction_1w INT,  -- 1 = up, 0 = flat, -1 = down
    direction_1m INT,
    direction_3m INT,
    direction_6m INT,
    -- Volatility-adjusted returns
    target_1w_volatility_adj DECIMAL(10, 6),
    target_1m_volatility_adj DECIMAL(10, 6),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

-- CRITICAL: targets must be computed at train time, not inference time
-- No look-ahead bias allowed

CREATE INDEX IF NOT EXISTS idx_features_targets_date 
    ON features.targets(date);

