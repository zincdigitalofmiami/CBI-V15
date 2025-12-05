#!/usr/bin/env python3
"""
Pull missing 1-hour and daily futures data from Databento Download Center
Identifies symbols we don't have and pulls both 1h and 1d data
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cbi_utils.keychain_manager import get_api_key
from src.cbi_utils.bigquery_client import get_client, load_dataframe_to_table, merge_staging_to_target

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
TABLE_1D = f"{PROJECT_ID}.raw.databento_futures_ohlcv_1d"
TABLE_1H = f"{PROJECT_ID}.raw.databento_futures_ohlcv_1h"

# Comprehensive list of futures symbols available on Databento GLBX.MDP3
# Based on CME/CBOT/NYMEX/COMEX futures
ALL_SYMBOLS = [
    # Agricultural (CBOT)
    'ZL.FUT',  # Soybean Oil
    'ZS.FUT',  # Soybeans
    'ZM.FUT',  # Soybean Meal
    'ZC.FUT',  # Corn
    'ZW.FUT',  # Wheat
    'ZO.FUT',  # Oats
    'KC.FUT',  # Coffee
    'CT.FUT',  # Cotton
    'SB.FUT',  # Sugar
    # 'CC.FUT',  # Cocoa - Not available as parent symbol, use specific contracts
    'LH.FUT',  # Lean Hogs
    'LC.FUT',  # Live Cattle
    'FC.FUT',  # Feeder Cattle
    
    # Energy (NYMEX)
    'CL.FUT',  # Crude Oil
    'HO.FUT',  # Heating Oil
    'RB.FUT',  # RBOB Gasoline
    'NG.FUT',  # Natural Gas
    
    # Metals (COMEX/NYMEX)
    'GC.FUT',  # Gold
    'SI.FUT',  # Silver
    'HG.FUT',  # Copper
    'PL.FUT',  # Platinum
    'PA.FUT',  # Palladium
    
    # Financials (CME)
    'ES.FUT',  # E-mini S&P 500
    'NQ.FUT',  # E-mini NASDAQ-100
    'YM.FUT',  # E-mini Dow
    'RTY.FUT',  # E-mini Russell 2000
    'ZN.FUT',  # 10-Year Treasury Note
    'ZB.FUT',  # 30-Year Treasury Bond
    'ZF.FUT',  # 5-Year Treasury Note
    'ZT.FUT',  # 2-Year Treasury Note
    '6E.FUT',  # Euro FX
    '6B.FUT',  # British Pound
    '6J.FUT',  # Japanese Yen
    '6A.FUT',  # Australian Dollar
    '6C.FUT',  # Canadian Dollar
    '6S.FUT',  # Swiss Franc
    '6N.FUT',  # New Zealand Dollar
]

def get_databento_client():
    """Initialize Databento client"""
    try:
        import databento as db
        api_key = get_api_key("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not found in Keychain")
        client = db.Historical(key=api_key)
        return client
    except ImportError:
        logger.error("databento package not installed. Install with: pip install databento")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Databento client: {e}")
        return None

def get_existing_symbols(table_id: str) -> set:
    """Get symbols we already have in BigQuery"""
    try:
        client = get_client()
        query = f"SELECT DISTINCT symbol FROM `{table_id}` ORDER BY symbol"
        result = client.query(query).to_dataframe()
        if result.empty:
            return set()
        return set(result['symbol'].unique())
    except Exception as e:
        logger.warning(f"Could not query existing symbols from {table_id}: {e}")
        return set()

def collect_data(symbol: str, schema: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Collect data for a symbol and schema (1h or 1d)
    
    Args:
        symbol: Futures symbol (e.g., 'ZL.FUT')
        schema: 'ohlcv-1h' or 'ohlcv-1d'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        
    Returns:
        DataFrame with columns matching table schema
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    client = get_databento_client()
    if not client:
        return pd.DataFrame()
    
    try:
        logger.info(f"Collecting {symbol} {schema} data from {start_date} to {end_date}")
        
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            stype_in="parent",
            schema=schema,
            start=start_date,
            end=end_date
        )
        
        records = []
        for record in data:
            if schema == 'ohlcv-1d':
                record_dict = {
                    'date': pd.to_datetime(record.pretty_ts_event).date(),
                    'open': float(record.pretty_open),
                    'high': float(record.pretty_high),
                    'low': float(record.pretty_low),
                    'close': float(record.pretty_close),
                    'volume': int(record.volume),
                    'open_interest': 0  # Not available in ohlcv-1d
                }
            else:  # ohlcv-1h
                # For 1h data, we need to extract hour from timestamp
                ts = pd.to_datetime(record.pretty_ts_event)
                record_dict = {
                    'date': ts.date(),
                    'hour': ts.hour,
                    'open': float(record.pretty_open),
                    'high': float(record.pretty_high),
                    'low': float(record.pretty_low),
                    'close': float(record.pretty_close),
                    'volume': int(record.volume),
                    'open_interest': 0
                }
            records.append(record_dict)
        
        if not records:
            logger.warning(f"No data returned for {symbol} {schema}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Extract root symbol (remove .FUT suffix)
        root_symbol = symbol.replace('.FUT', '')
        df['symbol'] = root_symbol
        
        # Select columns in EXACT order matching table schema
        if schema == 'ohlcv-1d':
            df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
            df['date'] = pd.to_datetime(df['date']).dt.date
        else:  # ohlcv-1h
            df = df[['date', 'hour', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['hour'] = df['hour'].astype(int)
        
        # Remove duplicates - keep last for same date (or date+hour for 1h)
        if schema == 'ohlcv-1d':
            df = df.drop_duplicates(subset=['symbol', 'date'], keep='last').sort_values(['symbol', 'date'])
        else:
            df = df.drop_duplicates(subset=['symbol', 'date', 'hour'], keep='last').sort_values(['symbol', 'date', 'hour'])
        
        # Ensure correct dtypes
        df['symbol'] = df['symbol'].astype(str)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(int)
        df['open_interest'] = df['open_interest'].astype(int)
        
        logger.info(f"Collected {len(df)} rows for {symbol} {schema}")
        return df
        
    except Exception as e:
        error_msg = str(e)
        if "symbology_invalid_request" in error_msg or "Could not resolve" in error_msg:
            logger.warning(f"Symbol {symbol} not available on Databento for {schema} - skipping")
        else:
            logger.error(f"Error collecting {symbol} {schema}: {e}")
            import traceback
            traceback.print_exc()
        return pd.DataFrame()

def get_last_date_for_symbol(table_id: str, symbol: str) -> str:
    """Get the last date we have data for a symbol"""
    try:
        client = get_client()
        query = f"""
        SELECT MAX(date) as last_date
        FROM `{table_id}`
        WHERE symbol = '{symbol}'
        """
        result = client.query(query).to_dataframe()
        if not result.empty and result['last_date'].iloc[0] is not None:
            last_date_val = result['last_date'].iloc[0]
            if pd.isna(last_date_val):
                return "2010-06-06"
            last_date = pd.to_datetime(last_date_val)
            if pd.isna(last_date):
                return "2010-06-06"
            return (last_date.date() + timedelta(days=1)).strftime("%Y-%m-%d")
        return "2010-06-06"  # Earliest Databento data
    except Exception as e:
        logger.warning(f"Could not determine last date for {symbol} from {table_id}: {e}")
        return "2010-06-06"

def ensure_table_exists(table_id: str, schema_type: str):
    """Ensure table exists, create if not"""
    try:
        client = get_client()
        client.get_table(table_id)
        logger.info(f"Table {table_id} exists")
    except Exception:
        logger.info(f"Creating table {table_id}...")
        if schema_type == '1d':
            create_sql = f"""
            CREATE TABLE `{table_id}` (
              date DATE,
              symbol STRING,
              open FLOAT64,
              high FLOAT64,
              low FLOAT64,
              close FLOAT64,
              volume INT64,
              open_interest INT64
            )
            PARTITION BY DATE_TRUNC(date, MONTH)
            CLUSTER BY symbol, date
            """
        else:  # 1h
            create_sql = f"""
            CREATE TABLE `{table_id}` (
              date DATE,
              hour INT64,
              symbol STRING,
              open FLOAT64,
              high FLOAT64,
              low FLOAT64,
              close FLOAT64,
              volume INT64,
              open_interest INT64
            )
            PARTITION BY DATE_TRUNC(date, MONTH)
            CLUSTER BY symbol, date, hour
            """
        client.query(create_sql).result()
        logger.info(f"✅ Created table {table_id}")

def main():
    """Main function to pull missing symbols"""
    logger.info("🚀 Starting Databento missing symbols collection")
    
    # Ensure tables exist
    ensure_table_exists(TABLE_1D, '1d')
    ensure_table_exists(TABLE_1H, '1h')
    
    # Get existing symbols from both tables
    existing_1d = get_existing_symbols(TABLE_1D)
    existing_1h = get_existing_symbols(TABLE_1H)
    
    logger.info(f"Existing 1d symbols: {sorted(existing_1d)}")
    logger.info(f"Existing 1h symbols: {sorted(existing_1h)}")
    
    # Convert ALL_SYMBOLS to root symbols (remove .FUT)
    all_root_symbols = {s.replace('.FUT', '') for s in ALL_SYMBOLS}
    
    # Find missing symbols
    missing_1d = all_root_symbols - existing_1d
    missing_1h = all_root_symbols - existing_1h
    
    logger.info(f"Missing 1d symbols: {sorted(missing_1d)}")
    logger.info(f"Missing 1h symbols: {sorted(missing_1h)}")
    
    if not missing_1d and not missing_1h:
        logger.info("✅ All symbols already have data!")
        return
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Collect missing 1d data
    if missing_1d:
        logger.info(f"📊 Collecting 1d data for {len(missing_1d)} missing symbols...")
        all_1d_data = []
        for symbol_root in sorted(missing_1d):
            symbol_fut = f"{symbol_root}.FUT"
            start_date = get_last_date_for_symbol(TABLE_1D, symbol_root)
            df = collect_data(symbol_fut, 'ohlcv-1d', start_date, end_date)
            if not df.empty:
                all_1d_data.append(df)
        
        if all_1d_data:
            combined_1d = pd.concat(all_1d_data, ignore_index=True)
            logger.info(f"Total 1d rows collected: {len(combined_1d)}")
            
            # Load into staging then MERGE
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            staging_table = f"{PROJECT_ID}.raw_staging.databento_1d_{run_id}"
            
            if load_dataframe_to_table(combined_1d, staging_table, PROJECT_ID, "WRITE_TRUNCATE"):
                if merge_staging_to_target(
                    staging_table=staging_table,
                    target_table=TABLE_1D,
                    key_columns=["symbol", "date"],
                    all_columns=["date", "symbol", "open", "high", "low", "close", "volume", "open_interest"],
                    project_id=PROJECT_ID,
                ):
                    logger.info(f"✅ Merged 1d data into {TABLE_1D}")
                else:
                    logger.error(f"❌ MERGE 1d data failed")
            else:
                logger.error(f"❌ Failed to load 1d data into staging")
    
    # Collect missing 1h data
    if missing_1h:
        logger.info(f"📊 Collecting 1h data for {len(missing_1h)} missing symbols...")
        all_1h_data = []
        for symbol_root in sorted(missing_1h):
            symbol_fut = f"{symbol_root}.FUT"
            start_date = get_last_date_for_symbol(TABLE_1H, symbol_root)
            df = collect_data(symbol_fut, 'ohlcv-1h', start_date, end_date)
            if not df.empty:
                all_1h_data.append(df)
        
        if all_1h_data:
            combined_1h = pd.concat(all_1h_data, ignore_index=True)
            logger.info(f"Total 1h rows collected: {len(combined_1h)}")
            
            # Load into staging then MERGE
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            staging_table = f"{PROJECT_ID}.raw_staging.databento_1h_{run_id}"
            
            if load_dataframe_to_table(combined_1h, staging_table, PROJECT_ID, "WRITE_TRUNCATE"):
                if merge_staging_to_target(
                    staging_table=staging_table,
                    target_table=TABLE_1H,
                    key_columns=["symbol", "date", "hour"],
                    all_columns=["date", "hour", "symbol", "open", "high", "low", "close", "volume", "open_interest"],
                    project_id=PROJECT_ID,
                ):
                    logger.info(f"✅ Merged 1h data into {TABLE_1H}")
                else:
                    logger.error(f"❌ MERGE 1h data failed")
            else:
                logger.error(f"❌ Failed to load 1h data into staging")
    
    logger.info("✅ Missing symbols collection complete!")

if __name__ == "__main__":
    main()

