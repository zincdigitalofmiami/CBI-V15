#!/usr/bin/env python3
"""
Pull additional futures data from Databento GLBX.MDP3
Includes: Livestock (LE, GF, HE), Dairy (CB, DC), FX (6M, 6L, 6Z, E7, J7)
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cbi_utils.keychain_manager import get_api_key
from src.cbi_utils.bigquery_client import get_client, load_dataframe_to_table, merge_staging_to_target

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
TABLE_1D = f"{PROJECT_ID}.raw.databento_futures_ohlcv_1d"
TABLE_1H = f"{PROJECT_ID}.raw.databento_futures_ohlcv_1h"

# Additional symbols to pull (verified available on GLBX.MDP3)
ADDITIONAL_SYMBOLS = [
    # Livestock (CME) - correct symbols
    'LE.FUT',  # Live Cattle
    'GF.FUT',  # Feeder Cattle
    'HE.FUT',  # Lean Hogs (alternative to LH)
    
    # Dairy (CME)
    'CB.FUT',  # Cash-settled Butter
    'DC.FUT',  # Class III Milk
    
    # Additional FX (CME)
    '6M.FUT',  # Mexican Peso
    '6L.FUT',  # Brazilian Real
    '6Z.FUT',  # South African Rand
    'E7.FUT',  # E-mini Euro
    'J7.FUT',  # E-mini Yen
]

def get_databento_client():
    """Initialize Databento client"""
    try:
        import databento as db
        api_key = get_api_key("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not found")
        return db.Historical(key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Databento client: {e}")
        return None

def get_existing_symbols(table_id: str) -> set:
    """Get symbols we already have"""
    try:
        client = get_client()
        query = f"SELECT DISTINCT symbol FROM `{table_id}`"
        result = client.query(query).to_dataframe()
        return set(result['symbol'].unique()) if not result.empty else set()
    except Exception as e:
        logger.warning(f"Could not query {table_id}: {e}")
        return set()

def collect_data(symbol: str, schema: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Collect OHLCV data for a symbol"""
    client = get_databento_client()
    if not client:
        return pd.DataFrame()
    
    try:
        logger.info(f"Collecting {symbol} {schema} from {start_date} to {end_date}")
        
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
                records.append({
                    'date': pd.to_datetime(record.pretty_ts_event).date(),
                    'open': float(record.pretty_open),
                    'high': float(record.pretty_high),
                    'low': float(record.pretty_low),
                    'close': float(record.pretty_close),
                    'volume': int(record.volume),
                    'open_interest': 0
                })
            else:  # ohlcv-1h
                ts = pd.to_datetime(record.pretty_ts_event)
                records.append({
                    'date': ts.date(),
                    'hour': ts.hour,
                    'open': float(record.pretty_open),
                    'high': float(record.pretty_high),
                    'low': float(record.pretty_low),
                    'close': float(record.pretty_close),
                    'volume': int(record.volume),
                    'open_interest': 0
                })
        
        if not records:
            logger.warning(f"No data for {symbol} {schema}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df['symbol'] = symbol.replace('.FUT', '')
        
        if schema == 'ohlcv-1d':
            df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
            df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
        else:
            df = df[['date', 'hour', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
            df = df.drop_duplicates(subset=['symbol', 'date', 'hour'], keep='last')
        
        logger.info(f"Collected {len(df)} rows for {symbol} {schema}")
        return df
        
    except Exception as e:
        if "symbology_invalid" in str(e) or "Could not resolve" in str(e):
            logger.warning(f"{symbol} not available - skipping")
        else:
            logger.error(f"Error collecting {symbol} {schema}: {e}")
        return pd.DataFrame()

def main():
    logger.info("🚀 Pulling additional Databento symbols")
    
    existing_1d = get_existing_symbols(TABLE_1D)
    existing_1h = get_existing_symbols(TABLE_1H)
    
    # Find symbols we don't have yet
    all_roots = {s.replace('.FUT', '') for s in ADDITIONAL_SYMBOLS}
    missing_1d = [f"{s}.FUT" for s in all_roots if s not in existing_1d]
    missing_1h = [f"{s}.FUT" for s in all_roots if s not in existing_1h]
    
    logger.info(f"Missing 1d: {[s.replace('.FUT','') for s in missing_1d]}")
    logger.info(f"Missing 1h: {[s.replace('.FUT','') for s in missing_1h]}")
    
    start_date = "2010-06-06"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Pull 1d data
    if missing_1d:
        all_1d = []
        for symbol in missing_1d:
            df = collect_data(symbol, 'ohlcv-1d', start_date, end_date)
            if not df.empty:
                all_1d.append(df)
        
        if all_1d:
            combined = pd.concat(all_1d, ignore_index=True)
            logger.info(f"Total 1d rows: {len(combined)}")
            
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            staging = f"{PROJECT_ID}.raw_staging.databento_add_1d_{run_id}"
            
            if load_dataframe_to_table(combined, staging, PROJECT_ID, "WRITE_TRUNCATE"):
                merge_staging_to_target(
                    staging_table=staging,
                    target_table=TABLE_1D,
                    key_columns=["symbol", "date"],
                    all_columns=["date", "symbol", "open", "high", "low", "close", "volume", "open_interest"],
                    project_id=PROJECT_ID
                )
                logger.info("✅ Merged 1d data")
    
    # Pull 1h data
    if missing_1h:
        all_1h = []
        for symbol in missing_1h:
            df = collect_data(symbol, 'ohlcv-1h', start_date, end_date)
            if not df.empty:
                all_1h.append(df)
        
        if all_1h:
            combined = pd.concat(all_1h, ignore_index=True)
            logger.info(f"Total 1h rows: {len(combined)}")
            
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            staging = f"{PROJECT_ID}.raw_staging.databento_add_1h_{run_id}"
            
            if load_dataframe_to_table(combined, staging, PROJECT_ID, "WRITE_TRUNCATE"):
                merge_staging_to_target(
                    staging_table=staging,
                    target_table=TABLE_1H,
                    key_columns=["symbol", "date", "hour"],
                    all_columns=["date", "hour", "symbol", "open", "high", "low", "close", "volume", "open_interest"],
                    project_id=PROJECT_ID
                )
                logger.info("✅ Merged 1h data")
    
    logger.info("✅ Additional symbols complete!")

if __name__ == "__main__":
    main()





