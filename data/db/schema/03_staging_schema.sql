-- MotherDuck Schema: staging
-- Normalized daily panels, AnoFox-cleaned data

-- Market daily panel (multi-symbol)
CREATE TABLE IF NOT EXISTS staging.market_daily (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    databento_open DECIMAL(10, 2),
    databento_high DECIMAL(10, 2),
    databento_low DECIMAL(10, 2),
    databento_close DECIMAL(10, 2),
    databento_volume BIGINT,
    databento_open_interest BIGINT,
    PRIMARY KEY (symbol, date)
);

-- ZL prices (AnoFox-cleaned)
CREATE TABLE IF NOT EXISTS staging.zl_prices_clean (
    date DATE PRIMARY KEY,
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT,
    open_interest BIGINT,
    gaps_filled INT DEFAULT 0,
    outliers_removed INT DEFAULT 0,
    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ZL realized volatility
CREATE TABLE IF NOT EXISTS staging.zl_realized_vol (
    date DATE PRIMARY KEY,
    realized_vol_5d DECIMAL(10, 6),
    realized_vol_21d DECIMAL(10, 6),
    realized_vol_63d DECIMAL(10, 6),
    garch_variance DECIMAL(10, 6),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FRED macro panel (wide format with prefixes)
CREATE TABLE IF NOT EXISTS staging.fred_macro_panel (
    date DATE PRIMARY KEY,
    fred_dxy DECIMAL(10, 4),
    fred_vix DECIMAL(10, 4),
    fred_treasury_10y DECIMAL(6, 4),
    fred_fed_funds DECIMAL(6, 4),
    fred_dtwexbgs DECIMAL(10, 4),
    fred_cpiaucsl DECIMAL(10, 4),
    -- (55+ FRED series as separate columns, all prefixed)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Weather granular daily (wide format by region)
CREATE TABLE IF NOT EXISTS staging.weather_granular_daily (
    date DATE PRIMARY KEY,
    weather_us_iowa_tavg_c DECIMAL(5, 2),
    weather_us_iowa_prcp_mm DECIMAL(6, 2),
    weather_us_illinois_tavg_c DECIMAL(5, 2),
    weather_br_mato_grosso_tavg_c DECIMAL(5, 2),
    weather_br_mato_grosso_prcp_mm DECIMAL(6, 2),
    weather_argentina_buenos_aires_tavg_c DECIMAL(5, 2),
    weather_argentina_drought_zscore DECIMAL(6, 4),
    -- (All key regions as separate columns)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- USDA reports granular (wide format)
CREATE TABLE IF NOT EXISTS staging.usda_reports_granular (
    date DATE PRIMARY KEY,
    usda_wasde_world_soyoil_prod DOUBLE,
    usda_wasde_us_soybean_yield DOUBLE,
    usda_exports_soybeans_net_sales_china DOUBLE,
    usda_cropprog_illinois_soybeans_condition_pct DECIMAL(5, 2),
    -- (All key USDA metrics as separate columns)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- EIA energy granular (wide format)
CREATE TABLE IF NOT EXISTS staging.eia_energy_granular (
    date DATE PRIMARY KEY,
    eia_biodiesel_prod_padd2 DOUBLE,
    eia_rin_price_d4 DECIMAL(10, 4),
    eia_rin_price_d6 DECIMAL(10, 4),
    eia_ethanol_prod_us DOUBLE,
    -- (All key EIA series as separate columns)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- CFTC commitments
CREATE TABLE IF NOT EXISTS staging.cftc_commitments (
    date DATE PRIMARY KEY,
    cftc_managed_money_netlong BIGINT,
    cftc_producer_merchant_short BIGINT,
    cftc_swap_dealer_long BIGINT,
    cftc_nonreportable_long BIGINT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Policy/Trump signals
CREATE TABLE IF NOT EXISTS staging.policy_trump_signals (
    date DATE PRIMARY KEY,
    policy_trump_score DECIMAL(5, 4),
    policy_trump_score_signed DECIMAL(6, 4),
    policy_trump_action_prob DECIMAL(5, 4),
    policy_trump_expected_zl_move DECIMAL(6, 4),
    geopolitical_tariff_score DECIMAL(5, 4),
    epa_rfs_event INT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

