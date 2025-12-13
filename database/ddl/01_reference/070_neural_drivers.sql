-- Neural Driver Groups (Big 8)
-- Maps driver groups to buckets for dashboard and model organization

CREATE TABLE IF NOT EXISTS reference.driver_group (
    driver_group_id VARCHAR PRIMARY KEY,
    driver_group_name VARCHAR NOT NULL,
    bucket_name VARCHAR NOT NULL,  -- Canonical: crush, china, fx, fed, tariff, biofuel, energy, volatility
    description TEXT,
    dashboard_page VARCHAR,
    update_frequency VARCHAR  -- '15min', 'daily', 'weekly'
);

-- Seed the 8 Big 8 driver groups
INSERT INTO reference.driver_group VALUES
('CRUSH', 'Crush Economics', 'crush', 'ZL/ZS/ZM spread economics, oil share, board crush', 'Dashboard', 'daily'),
('CHINA', 'China Demand', 'china', 'China demand proxy (HG copper, export sales)', 'Trade Intelligence', 'daily'),
('FX', 'Currency Effects', 'fx', 'Currency effects (DX, BRL, CNY, MXN)', 'Dashboard', '15min'),
('FED', 'Monetary Policy', 'fed', 'Fed funds, yield curve, NFCI', 'Dashboard', 'daily'),
('TARIFF', 'Trade Policy', 'tariff', 'Trade policy (Trump sentiment, Section 301)', 'Trade Intelligence', '15min'),
('BIOFUEL', 'Biofuel Markets', 'biofuel', 'RIN prices, biodiesel, RFS mandates, BOHO spread', 'Strategy', 'daily'),
('ENERGY', 'Energy Complex', 'energy', 'Crude, HO, RB, crack spreads', 'Dashboard', '15min'),
('VOLATILITY', 'Market Volatility', 'volatility', 'VIX, realized vol, STLFSI4, stress indices', 'Dashboard', '15min')
ON CONFLICT DO NOTHING;

-- Feature to driver group mapping
CREATE TABLE IF NOT EXISTS reference.feature_to_driver_group_map (
    feature_name VARCHAR PRIMARY KEY,
    driver_group_id VARCHAR NOT NULL REFERENCES reference.driver_group(driver_group_id),
    feature_description TEXT,
    data_source VARCHAR,
    mapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

