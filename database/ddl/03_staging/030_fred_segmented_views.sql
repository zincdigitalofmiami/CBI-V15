-- ============================================================================
-- FRED Data Segmentation Views - Organized by Big 8 Buckets
-- ============================================================================
-- Purpose: Segment raw.fred_economic into clean views by bucket
-- Makes feature engineering easier and clearer
-- ============================================================================

-- ============================================================================
-- BUCKET 4: FED (Monetary Policy & Interest Rates)
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_fed_policy AS
SELECT 
    date as as_of_date,
    series_id,
    value,
    'fed' as bucket
FROM raw.fred_economic
WHERE series_id IN (
    -- Fed Funds
    'DFF',           -- Fed Funds Effective Rate (1954-present)
    'DFEDTARU',      -- Fed Funds Target Upper (2008-present)
    'DFEDTARL',      -- Fed Funds Target Lower (2008-present)
    'SOFR',          -- Secured Overnight Financing Rate (2018-present)
    'IORB',          -- Interest on Reserve Balances (2021-present)
    
    -- Treasury Yields
    'DGS10',         -- 10-Year Treasury (1962-present)
    'DGS2',          -- 2-Year Treasury (1976-present)
    'DGS5',          -- 5-Year Treasury
    'DGS30',         -- 30-Year Treasury
    'T10Y2Y',        -- 10Y-2Y Spread (yield curve)
    'T10Y3M',        -- 10Y-3M Spread
    
    -- Other Rates
    'MORTGAGE30US',  -- 30-Year Mortgage Rate
    'DPRIME',        -- Bank Prime Loan Rate
    'FEDFUNDS'       -- Federal Funds Rate (alternative)
)
ORDER BY as_of_date, series_id;

-- ============================================================================
-- BUCKET 8: VOLATILITY (Financial Stress & VIX)
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_volatility AS
SELECT 
    date as as_of_date,
    series_id,
    value,
    'volatility' as bucket
FROM raw.fred_economic
WHERE series_id IN (
    -- Volatility Indices
    'VIXCLS',        -- VIX (1990-present)
    'VXVCLS',        -- VIX 3-Month
    'VXDCLS',        -- CBOE DJIA Volatility
    
    -- Financial Stress
    'STLFSI',        -- St. Louis Fed Financial Stress (1993-present)
    'STLFSI4',       -- STLFSI 4-Week MA (1993-present)
    'NFCI',          -- Chicago Fed National Financial Conditions (1971-present)
    'NFCILEVERAGE',  -- NFCI Leverage Subindex
    'NFCICREDIT',    -- NFCI Credit Subindex
    'NFCINONFINLEVERAGE',  -- NFCI Nonfinancial Leverage
    
    -- Credit Spreads
    'BAMLH0A0HYM2',  -- High Yield Spread
    'BAMLC0A0CM',    -- Corporate Bond Spread
    'T10Y3M'         -- Term spread (also recession indicator)
)
ORDER BY as_of_date, series_id;

-- ============================================================================
-- BUCKET 3: FX (Foreign Exchange Rates)
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_fx AS
SELECT 
    date as as_of_date,
    series_id,
    value,
    'fx' as bucket
FROM raw.fred_economic
WHERE series_id IN (
    -- Major FX Rates (Daily)
    'DEXBZUS',       -- Brazil Real (1995-present) - CRITICAL for ZL
    'DEXCHUS',       -- China Yuan (1981-present) - CRITICAL for China bucket
    'DEXMXUS',       -- Mexico Peso (1993-present)
    'DEXUSEU',       -- Euro (1999-present)
    'DEXJPUS',       -- Japanese Yen
    'DEXUSUK',       -- British Pound
    'DEXCAUS',       -- Canadian Dollar
    
    -- Dollar Indices
    'DTWEXBGS',      -- Dollar Index Broad (2006-present)
    'DTWEXM',        -- Dollar Index Major Currencies
    'DTWEXEMEGS',    -- Dollar Index Emerging Markets
    'DTWEXAFEGS',    -- Dollar Index Advanced Foreign Economies
    
    -- Alternative FX
    'DEXUSAL',       -- Australian Dollar
    'DEXKOUS',       -- South Korean Won
    'DEXINUS',       -- Indian Rupee
    'DEXSFUS'        -- South African Rand
)
ORDER BY as_of_date, series_id;

-- ============================================================================
-- BUCKET 7: ENERGY (Oil, Gas, Petroleum Products)
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_energy AS
SELECT 
    date as as_of_date,
    series_id,
    value,
    'energy' as bucket
FROM raw.fred_economic
WHERE series_id IN (
    -- Crude Oil (Daily)
    'DCOILWTICO',    -- WTI Crude (1986-present)
    'DCOILBRENTEU',  -- Brent Crude
    'MCOILWTICO',    -- WTI Monthly
    
    -- Natural Gas (Daily)
    'DHHNGSP',       -- Henry Hub Natural Gas (1997-present)
    'MHHNGSP',       -- Natural Gas Monthly
    
    -- Petroleum Products (Weekly)
    'GASREGW',       -- Regular Gasoline (1990-present)
    'GASDESW',       -- Diesel (1994-present)
    'DHOILNYH',      -- Heating Oil NY Harbor (1986-present)
    
    -- Energy Indices
    'PNRGINDEXM',    -- Energy Price Index
    'PPIACO'         -- Producer Price Index: Commodities
)
ORDER BY as_of_date, series_id;

-- ============================================================================
-- BUCKET 1: CRUSH (Agriculture Prices - Monthly)
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_agriculture AS
SELECT 
    date as as_of_date,
    series_id,
    value,
    'crush' as bucket
FROM raw.fred_economic
WHERE series_id IN (
    -- Soybeans
    'PSOYBUSDM',     -- Global Soybean Price (1990-present)
    'PSOYBUSDQ',     -- Quarterly
    'PSOYBUSDM',     -- Monthly
    
    -- Corn
    'PMAIZMTUSDM',   -- Global Corn Price (1990-present)
    'PMAIZMTUSDA',   -- Annual
    
    -- Wheat
    'PWHEAMTUSDM',   -- Global Wheat Price (1990-present)
    
    -- Other Ag
    'PRICENPQ',      -- Rice Price
    'PCOTTINDUSDM'   -- Cotton Price
)
ORDER BY as_of_date, series_id;

-- ============================================================================
-- BUCKET 2: CHINA (Copper as Proxy)
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_china_proxy AS
SELECT 
    date as as_of_date,
    series_id,
    value,
    'china' as bucket
FROM raw.fred_economic
WHERE series_id IN (
    -- Copper (China demand proxy)
    'PCOPPUSDM',     -- Global Copper Price (1990-present)
    'PCOPPUSDQ',     -- Quarterly
    
    -- China-specific
    'CHNRGDPEXP',    -- China GDP Growth
    'CHNPIEATI01GYM' -- China CPI
)
ORDER BY as_of_date, series_id;

-- ============================================================================
-- MACRO CONTEXT (CPI, Employment, GDP)
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_macro AS
SELECT 
    date as as_of_date,
    series_id,
    value,
    'macro' as bucket
FROM raw.fred_economic
WHERE series_id IN (
    -- Inflation
    'CPIAUCSL',      -- CPI (1947-present)
    'CPILFESL',      -- Core CPI
    'PCEPI',         -- PCE (1959-present)
    'PCEPILFE',      -- Core PCE
    
    -- Employment
    'UNRATE',        -- Unemployment Rate (1948-present)
    'PAYEMS',        -- Nonfarm Payrolls
    'CIVPART',       -- Labor Force Participation
    
    -- GDP
    'GDP',           -- Gross Domestic Product
    'GDPC1',         -- Real GDP
    'GDPPOT'         -- Potential GDP
)
ORDER BY as_of_date, series_id;

-- ============================================================================
-- EQUITY INDICES (Risk-On/Risk-Off Sentiment)
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_equity_indices AS
SELECT 
    date as as_of_date,
    series_id,
    value,
    'volatility' as bucket
FROM raw.fred_economic
WHERE series_id IN (
    -- Modern Indices
    'SP500',         -- S&P 500 (2015-present)
    'NASDAQCOM',     -- NASDAQ (1971-present)
    
    -- Historical Indices
    'SP500_HISTORICAL',      -- Combined 1871-2025
    'NYSE_COMPOSITE',        -- NYSE 1945-present
    'M1125AUSM343NNBR',      -- Cowles Commission (1871-1956)
    'BOGZ1FL073164003Q'      -- NYSE Quarterly (1945-present)
)
ORDER BY as_of_date, series_id;

-- ============================================================================
-- MASTER VIEW: All FRED Data with Bucket Labels
-- ============================================================================
CREATE OR REPLACE VIEW staging.fred_all_segmented AS
SELECT * FROM staging.fred_fed_policy
UNION ALL SELECT * FROM staging.fred_volatility
UNION ALL SELECT * FROM staging.fred_fx
UNION ALL SELECT * FROM staging.fred_energy
UNION ALL SELECT * FROM staging.fred_agriculture
UNION ALL SELECT * FROM staging.fred_china_proxy
UNION ALL SELECT * FROM staging.fred_macro
UNION ALL SELECT * FROM staging.fred_equity_indices;


