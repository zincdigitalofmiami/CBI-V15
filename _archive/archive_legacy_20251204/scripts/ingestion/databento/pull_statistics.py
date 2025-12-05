#!/usr/bin/env python3
"""
Pull statistics data from Databento GLBX.MDP3 for all symbols
Includes: Settlement prices, Open Interest, Session High/Low, etc.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import logging

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cbi_utils.keychain_manager import get_api_key
from src.cbi_utils.bigquery_client import get_client, load_dataframe_to_table, merge_staging_to_target

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
TABLE_STATS = f"{PROJECT_ID}.raw.databento_futures_statistics"

# All symbols we have in daily data
ALL_SYMBOLS = [
    '6A.FUT', '6B.FUT', '6C.FUT', '6E.FUT', '6J.FUT', '6L.FUT', '6M.FUT', 
    '6N.FUT', '6S.FUT', '6Z.FUT', 'CB.FUT', 'CL.FUT', 'DC.FUT', 'E7.FUT', 
    'ES.FUT', 'GC.FUT', 'GF.FUT', 'HE.FUT', 'HG.FUT', 'HO.FUT', 'J7.FUT', 
    'LE.FUT', 'LH.FUT', 'NG.FUT', 'NQ.FUT', 'PA.FUT', 'PL.FUT', 'RB.FUT', 
    'RTY.FUT', 'SI.FUT', 'YM.FUT', 'ZB.FUT', 'ZC.FUT', 'ZF.FUT', 'ZL.FUT', 
    'ZM.FUT', 'ZN.FUT', 'ZO.FUT', 'ZS.FUT', 'ZT.FUT', 'ZW.FUT'
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

def collect_statistics(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Collect statistics data for a symbol"""
    client = get_databento_client()
    if not client:
        return pd.DataFrame()
    
    try:
        logger.info(f"Collecting {symbol} statistics from {start_date} to {end_date}")
        
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            stype_in="parent",
            schema="statistics",
            start=start_date,
            end=end_date
        )
        
        records = []
        seen = set()  # Track unique date/stat_type combos
        
        for record in data:
            ts_event = pd.to_datetime(record.ts_event, unit='ns', utc=True)
            ts_recv = pd.to_datetime(record.ts_recv, unit='ns', utc=True)
            
            # Convert price (fixed point to float)
            price = record.pretty_price if hasattr(record, 'pretty_price') else record.price / 1e9
            
            # Skip invalid quantities
            quantity = record.quantity if record.quantity < 9223372036854775807 else None
            
            stat_type = str(record.stat_type).replace('StatType.', '').replace('<', '').replace('>', '').split(':')[0]
            date_val = ts_event.date()
            
            # Only keep one record per date/stat_type (the latest)
            key = (date_val, stat_type)
            if key in seen:
                continue
            seen.add(key)
            
            records.append({
                'date': date_val,
                'symbol': symbol.replace('.FUT', ''),
                'stat_type': stat_type,
                'price': float(price) if price else None,
                'quantity': int(quantity) if quantity else None,
                'ts_event': ts_event,
                'ts_recv': ts_recv,
                'update_action': str(record.update_action).replace('StatUpdateAction.', '').replace('<', '').replace('>', '').split(':')[0]
            })
        
        if not records:
            logger.warning(f"No statistics for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Keep only the latest record per date/symbol/stat_type
        df = df.sort_values('ts_event').drop_duplicates(
            subset=['date', 'symbol', 'stat_type'], 
            keep='last'
        )
        
        logger.info(f"Collected {len(df)} statistics records for {symbol}")
        return df
        
    except Exception as e:
        if "symbology_invalid" in str(e):
            logger.warning(f"{symbol} not available - skipping")
        else:
            logger.error(f"Error collecting {symbol} statistics: {e}")
        return pd.DataFrame()

def main():
    logger.info("🚀 Pulling Databento statistics for all symbols")
    
    # Pull last 30 days of statistics (statistics are very large)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    all_stats = []
    for symbol in ALL_SYMBOLS:
        df = collect_statistics(symbol, start_date, end_date)
        if not df.empty:
            all_stats.append(df)
    
    if not all_stats:
        logger.warning("No statistics collected")
        return
    
    combined = pd.concat(all_stats, ignore_index=True)
    logger.info(f"Total statistics rows: {len(combined)}")
    
    # Load to BigQuery
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    staging = f"{PROJECT_ID}.raw_staging.databento_stats_{run_id}"
    
    if load_dataframe_to_table(combined, staging, PROJECT_ID, "WRITE_TRUNCATE"):
        merge_staging_to_target(
            staging_table=staging,
            target_table=TABLE_STATS,
            key_columns=["symbol", "date", "stat_type"],
            all_columns=["date", "symbol", "stat_type", "price", "quantity", "ts_event", "ts_recv", "update_action"],
            project_id=PROJECT_ID
        )
        logger.info("✅ Statistics merged successfully")
    
    # Print summary
    print("\n=== STATISTICS SUMMARY ===")
    summary = combined.groupby('stat_type').agg({
        'symbol': 'nunique',
        'date': 'nunique',
        'price': 'count'
    }).rename(columns={'symbol': 'symbols', 'date': 'dates', 'price': 'records'})
    print(summary.to_string())

if __name__ == "__main__":
    main()

