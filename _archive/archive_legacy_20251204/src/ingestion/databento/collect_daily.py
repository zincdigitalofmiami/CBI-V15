#!/usr/bin/env python3
"""
Collect daily OHLCV futures data from Databento API
Stores to: raw.databento_futures_ohlcv_1d
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging
from google.cloud import bigquery

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cbi_utils.keychain_manager import get_api_key
from src.cbi_utils.bigquery_client import get_client, load_dataframe_to_table, merge_staging_to_target

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
TABLE_ID = f"{PROJECT_ID}.raw.databento_futures_ohlcv_1d"

# Symbols to collect - Databento requires .FUT suffix for parent symbols
# Core ag/energy/metals panel:
#   grains/soy: ZL, ZS, ZM, ZC
#   energy: CL (crude), HO (heating oil), RB (gasoline)
#   metals: GC (gold), SI (silver), HG (copper)
#   livestock: HE (lean hog)
SYMBOLS = [
    'ZL.FUT', 'ZS.FUT', 'ZM.FUT', 'ZC.FUT',
    'CL.FUT', 'HO.FUT', 'RB.FUT',
    'GC.FUT', 'SI.FUT', 'HG.FUT',
    'HE.FUT',
]

def get_databento_client():
    """Initialize Databento client"""
    try:
        import databento as db
        api_key = get_api_key("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not found in Keychain. Run store_api_keys.sh")
        client = db.Historical(key=api_key)
        return client
    except ImportError:
        logger.error("databento package not installed. Install with: pip install databento")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Databento client: {e}")
        return None

def collect_daily_data(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Collect daily OHLCV data for a symbol
    
    Args:
        symbol: Futures symbol (e.g., 'ZL')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        
    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume, open_interest
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    client = get_databento_client()
    if not client:
        return pd.DataFrame()
    
    try:
        logger.info(f"Collecting {symbol} data from {start_date} to {end_date}")
        
        # Databento API call - returns DBNStore iterator
        # Use stype_in='parent' to get continuous contracts
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            stype_in="parent",
            schema="ohlcv-1d",
            start=start_date,
            end=end_date
        )
        
        # Convert DBNStore to DataFrame - extract fields from OhlcVMsg objects
        # Use pretty_* attributes for easier conversion
        # MUST match existing table schema: date, symbol, open, high, low, close, volume, open_interest
        records = []
        try:
            for record in data:
                # Extract fields from OhlcVMsg object (use pretty_* for pre-converted values)
                record_dict = {
                    'date': pd.to_datetime(record.pretty_ts_event).date(),
                    'open': float(record.pretty_open),
                    'high': float(record.pretty_high),
                    'low': float(record.pretty_low),
                    'close': float(record.pretty_close),
                    'volume': int(record.volume),
                    'open_interest': 0  # Not available in ohlcv-1d schema, set to 0
                }
                records.append(record_dict)
        except Exception as e:
            logger.error(f"Error iterating records for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        
        if not records:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Extract root symbol (remove .FUT suffix for storage)
        root_symbol = symbol.replace('.FUT', '')
        df['symbol'] = root_symbol
        
        # Select columns in EXACT order matching table schema
        df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
        
        # Remove duplicates by date (keep last)
        df = df.drop_duplicates(subset=['date'], keep='last').sort_values('date')
        
        # Ensure correct dtypes matching table schema
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['symbol'] = df['symbol'].astype(str)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(int)
        df['open_interest'] = df['open_interest'].astype(int)
        
        logger.info(f"Collected {len(df)} rows for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error collecting {symbol}: {e}")
        return pd.DataFrame()

def get_last_date_for_symbol(client, root_symbol: str):
    """Return last loaded date for a given root symbol, or None if empty."""
    query = f"""
    SELECT MAX(date) as last_date
    FROM `{TABLE_ID}`
    WHERE symbol = @symbol
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("symbol", "STRING", root_symbol)
        ]
    )
    try:
        result = client.query(query, job_config=job_config).to_dataframe()
        if not result.empty:
            last = result["last_date"].iloc[0]
            if pd.notna(last):
                return pd.to_datetime(last).date()
    except Exception as e:
        logger.warning(f"Could not determine last date for {root_symbol}: {e}")
    return None


def main():
    """Main ingestion function"""
    logger.info("🚀 Starting Databento daily data collection")

    client = get_client()
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_data = []
    for symbol in SYMBOLS:
        root_symbol = symbol.replace(".FUT", "")
        last_date = get_last_date_for_symbol(client, root_symbol)
        if last_date is not None:
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"{root_symbol}: resuming from {start_date}")
        else:
            start_date = "2010-06-06"
            logger.info(f"{root_symbol}: no existing data, starting from {start_date}")

        df = collect_daily_data(symbol, start_date, end_date)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        logger.warning("No data collected")
        return
    
    # Combine all symbols
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total rows collected this run: {len(combined_df)}")
    
    # Load into staging then MERGE into canonical raw.databento_futures_ohlcv_1d
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    staging_dataset = f"{PROJECT_ID}.raw_staging"
    staging_table = f"{staging_dataset}.databento_daily_{run_id}"
    
    client = get_client()
    try:
        client.get_dataset(staging_dataset)
    except Exception:
        from google.cloud import bigquery as bq
        client.create_dataset(bq.Dataset(staging_dataset))
    
    if not load_dataframe_to_table(combined_df, staging_table, PROJECT_ID, "WRITE_TRUNCATE"):
        logger.error(f"❌ Failed to load Databento data into staging table {staging_table}")
        return
    
    if merge_staging_to_target(
        staging_table=staging_table,
        target_table=TABLE_ID,
        key_columns=["symbol", "date"],
        all_columns=["date", "symbol", "open", "high", "low", "close", "volume", "open_interest"],
        project_id=PROJECT_ID,
    ):
        logger.info(f"✅ Merged Databento data from {staging_table} into {TABLE_ID}")
    else:
        logger.error(f"❌ MERGE from {staging_table} into {TABLE_ID} failed")

if __name__ == "__main__":
    main()
