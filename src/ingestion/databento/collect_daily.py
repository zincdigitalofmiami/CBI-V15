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

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.keychain_manager import get_api_key
from src.utils.bigquery_client import get_client, load_dataframe_to_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
TABLE_ID = f"{PROJECT_ID}.raw.databento_futures_ohlcv_1d"

# Symbols to collect (ZL-focused)
SYMBOLS = ['ZL', 'ZS', 'ZM', 'CL', 'HO', 'FCPO']

def get_databento_client():
    """Initialize Databento client"""
    try:
        import databento as db
        api_key = get_api_key("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not found in Keychain. Run store_api_keys.sh")
        client = db.Historical(api_key=api_key)
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
        
        # Databento API call (adjust based on actual API)
        # This is a template - adjust to actual Databento API
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            schema="ohlcv-1d",
            start=start_date,
            end=end_date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Standardize column names
        df = df.rename(columns={
            'ts_event': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'open_interest': 'open_interest'
        })
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Ensure date is DATE type
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Select columns in order
        df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
        
        logger.info(f"Collected {len(df)} rows for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error collecting {symbol}: {e}")
        return pd.DataFrame()

def main():
    """Main ingestion function"""
    logger.info("🚀 Starting Databento daily data collection")
    
    # Get last date from BigQuery
    client = get_client()
    try:
        query = f"""
        SELECT MAX(date) as last_date
        FROM `{TABLE_ID}`
        WHERE symbol = 'ZL'
        """
        result = client.query(query).to_dataframe()
        if not result.empty and result['last_date'].iloc[0] is not None:
            last_date = pd.to_datetime(result['last_date'].iloc[0]).date()
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"Resuming from {start_date}")
        else:
            # Start from 2010 if no data exists
            start_date = "2010-01-01"
            logger.info(f"No existing data, starting from {start_date}")
    except Exception as e:
        logger.warning(f"Could not determine last date: {e}. Starting from 2010-01-01")
        start_date = "2010-01-01"
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Collect data for all symbols
    all_data = []
    for symbol in SYMBOLS:
        df = collect_daily_data(symbol, start_date, end_date)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        logger.warning("No data collected")
        return
    
    # Combine all symbols
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total rows collected: {len(combined_df)}")
    
    # Load to BigQuery
    if load_dataframe_to_table(combined_df, TABLE_ID, PROJECT_ID, "WRITE_APPEND"):
        logger.info(f"✅ Successfully loaded {len(combined_df)} rows to {TABLE_ID}")
    else:
        logger.error("❌ Failed to load data to BigQuery")

if __name__ == "__main__":
    main()

