#!/usr/bin/env python3
"""
Databento to DuckDB Pipeline - Phase 2.4
Ingests ZL futures data from Databento and loads into DuckDB using COPY statements.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging
import duckdb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cbi_utils.keychain_manager import get_api_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DUCKDB_PATH = Path('/Volumes/Satechi Hub/ZL-Intelligence/duckdb/cbi-v15.duckdb')
PARQUET_CACHE = Path('/Volumes/Satechi Hub/ZL-Intelligence/parquet/raw/databento')

# ZL futures symbol
SYMBOL = 'ZL.FUT'


def get_databento_client():
    """Initialize Databento client."""
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


def fetch_databento_data(client, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch data from Databento API."""
    try:
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            schema="ohlcv-1d",
            start=start_date,
            end=end_date
        )
        
        # Convert to DataFrame
        df = data.to_df()
        
        if df.empty:
            logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
            return pd.DataFrame()
        
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
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Add symbol column
        df['symbol'] = symbol.replace('.FUT', '')
        
        # Select and order columns
        cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
        df = df[[c for c in cols if c in df.columns]]
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Databento data: {e}")
        return pd.DataFrame()


def save_to_parquet(df: pd.DataFrame, output_path: Path):
    """Save DataFrame to Parquet with proper types."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure proper types
    df['date'] = pd.to_datetime(df['date'])
    
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )


def load_to_duckdb(conn: duckdb.DuckDBPyConnection, parquet_file: Path, table_name: str):
    """Load Parquet file into DuckDB using COPY."""
    try:
        # Drop table if exists (for clean reload)
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table from Parquet
        conn.execute(f"""
            CREATE TABLE {table_name} AS 
            SELECT * FROM read_parquet('{parquet_file}')
        """)
        
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Create indexes
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name.replace('.', '_')}_date ON {table_name}(date)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name.replace('.', '_')}_symbol ON {table_name}(symbol)")
        except:
            pass
        
        return row_count
        
    except Exception as e:
        logger.error(f"Error loading to DuckDB: {e}")
        raise


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("DATABENTO TO DUCKDB PIPELINE - Phase 2.4")
    print("=" * 60)
    print(f"Symbol: {SYMBOL}")
    print(f"DuckDB: {DUCKDB_PATH}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Initialize Databento client
    client = get_databento_client()
    if not client:
        return
    
    # Initialize DuckDB
    if not DUCKDB_PATH.exists():
        logger.error(f"DuckDB database not found at {DUCKDB_PATH}")
        logger.error("Run create_duckdb_schema.py first!")
        return
    
    conn = duckdb.connect(str(DUCKDB_PATH))
    
    try:
        # Determine date range
        # Check existing data in DuckDB
        try:
            existing = conn.execute("SELECT MAX(date) as max_date FROM raw.zl_fact_prices_features").fetchone()
            if existing and existing[0]:
                start_date = pd.to_datetime(existing[0]).date() + timedelta(days=1)
                print(f"Found existing data up to: {existing[0]}")
                print(f"Starting from: {start_date}")
            else:
                # Start from 2000-01-01 for full backfill
                start_date = datetime(2000, 1, 1).date()
                print("No existing data, starting full backfill from 2000-01-01")
        except:
            start_date = datetime(2000, 1, 1).date()
            print("No existing table, starting full backfill from 2000-01-01")
        
        end_date = datetime.now().date()
        
        # Fetch data from Databento
        print(f"Fetching data from Databento: {start_date} to {end_date}")
        df = fetch_databento_data(client, SYMBOL, start_date, end_date)
        
        if df.empty:
            print("No new data to ingest")
            return
        
        print(f"Fetched {len(df):,} rows")
        
        # Save to Parquet cache
        parquet_file = PARQUET_CACHE / f"zl_futures_{start_date}_{end_date}.parquet"
        print(f"Saving to Parquet: {parquet_file}")
        save_to_parquet(df, parquet_file)
        
        # Load to DuckDB
        table_name = "raw.zl_fact_prices_features"
        print(f"Loading to DuckDB table: {table_name}")
        row_count = load_to_duckdb(conn, parquet_file, table_name)
        
        print()
        print("=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Rows loaded: {row_count:,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Table: {table_name}")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()

