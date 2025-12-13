#!/usr/bin/env python3
"""Quick verification that big8_bucket_features.sql loads without errors.

Creates stub tables matching DDL structure to allow macro binding validation.
"""
import duckdb
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

con = duckdb.connect(':memory:')

# Create schemas
con.execute('CREATE SCHEMA IF NOT EXISTS raw')
con.execute('CREATE SCHEMA IF NOT EXISTS staging')

# Databento raw table (matches database/ddl/02_raw/010_raw_databento_ohlcv.sql)
con.execute('''
    CREATE TABLE raw.databento_futures_ohlcv_1d (
        symbol TEXT, 
        as_of_date DATE, 
        open DOUBLE, 
        high DOUBLE, 
        low DOUBLE, 
        close DOUBLE, 
        volume BIGINT, 
        open_interest BIGINT
    )
''')

# CFTC COT table (matches database/ddl/02_raw/090_raw_cftc_cot.sql)
con.execute('''
    CREATE TABLE raw.cftc_cot (
        report_date DATE, 
        symbol TEXT,
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
        prod_merc_net BIGINT,
        swap_net BIGINT,
        managed_money_net BIGINT,
        other_rept_net BIGINT,
        nonrept_net BIGINT,
        managed_money_net_pct_oi DOUBLE, 
        prod_merc_net_pct_oi DOUBLE
    )
''')

# CFTC COT disaggregated view (matches database/ddl/02_raw/095_raw_cftc_cot_views.sql)
con.execute('''CREATE VIEW raw.cftc_cot_disaggregated AS SELECT * FROM raw.cftc_cot''')

# CFTC TFF table (matches database/ddl/02_raw/090_raw_cftc_cot.sql)
con.execute('''
    CREATE TABLE raw.cftc_cot_tff (
        report_date DATE, 
        symbol TEXT,
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
        dealer_net BIGINT,
        asset_mgr_net BIGINT,
        lev_money_net BIGINT,
        other_rept_net BIGINT,
        nonrept_net BIGINT,
        lev_money_net_pct_oi DOUBLE,
        asset_mgr_net_pct_oi DOUBLE
    )
''')

# EIA Biofuels (matches database/ddl/02_raw/060_raw_eia_biofuels.sql)
con.execute('CREATE TABLE raw.eia_biofuels (date DATE, series_id TEXT, value DOUBLE)')

# EPA RIN prices (matches database/ddl/02_raw/070_raw_epa_rin_prices.sql)
con.execute('CREATE TABLE raw.epa_rin_prices (date DATE, series_id TEXT, value DOUBLE)')

# FRED Economic (matches database/ddl/02_raw/020_raw_fred_economic.sql)
con.execute('CREATE TABLE raw.fred_economic (date DATE, series_id TEXT, value DOUBLE)')

# ScrapeCreators News (matches database/ddl/02_raw/080_raw_news_articles.sql)
con.execute('''
    CREATE TABLE raw.scrapecreators_news_buckets (
        date DATE, 
        bucket VARCHAR,
        headline TEXT,
        zl_sentiment DOUBLE,
        is_trump_related BOOLEAN, 
        policy_axis TEXT
    )
''')

# Staging CFTC normalized (matches database/ddl/03_staging/070_staging_cftc_normalized.sql)
con.execute('''
    CREATE TABLE staging.cftc_normalized (
        date DATE,
        commodity TEXT,
        managed_money_net BIGINT,
        producer_merchant_net BIGINT,
        swap_dealer_net BIGINT,
        managed_money_net_pctile DOUBLE,
        producer_merchant_net_pctile DOUBLE,
        total_open_interest BIGINT
    )
''')

# Load big8 macro
macro_file = REPO_ROOT / 'database/macros/big8_bucket_features.sql'
with open(macro_file) as f:
    sql = f.read()
    try:
        con.execute(sql)
        print('SUCCESS: big8_bucket_features.sql loads without errors')
    except Exception as e:
        print(f'FAILED: {e}')
        raise

# Load COT enhancements macro
cot_macro_file = REPO_ROOT / 'database/macros/big8_cot_enhancements.sql'
with open(cot_macro_file) as f:
    sql = f.read()
    try:
        con.execute(sql)
        print('SUCCESS: big8_cot_enhancements.sql loads without errors')
    except Exception as e:
        print(f'FAILED: {e}')
        raise

print('\n✅ All macros validated successfully')
