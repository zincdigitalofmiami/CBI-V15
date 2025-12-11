#!/usr/bin/env python3
"""Quick verification that big8_bucket_features.sql loads without errors."""
import duckdb
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

con = duckdb.connect(':memory:')

# Create schemas
con.execute('CREATE SCHEMA IF NOT EXISTS raw')

# Create test stubs
con.execute('CREATE TABLE raw.databento_ohlcv_daily (as_of_date DATE, symbol TEXT, close DOUBLE, high DOUBLE, low DOUBLE, volume DOUBLE)')
con.execute('CREATE TABLE raw.cftc_cot_disaggregated (report_date DATE, symbol TEXT, managed_money_net_pct_oi DOUBLE, prod_merc_net_pct_oi DOUBLE)')
con.execute('CREATE TABLE raw.eia_biofuels (date DATE, series_id TEXT, value DOUBLE)')
con.execute('CREATE TABLE raw.epa_rin_prices (date DATE, series_id TEXT, value DOUBLE)')
con.execute('CREATE TABLE raw.fred_observations (date DATE, series_id TEXT, value DOUBLE)')
con.execute('CREATE TABLE raw.scrapecreators_news_buckets (date DATE, zl_sentiment TEXT, is_trump_related BOOLEAN, policy_axis TEXT)')

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

