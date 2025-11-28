#!/usr/bin/env python3
"""
Technical Indicators Calculation - Incremental Updates
Uses pandas-ta for daily updates
"""

import pandas as pd
import pandas_ta as ta
from google.cloud import bigquery
from datetime import datetime, timedelta

PROJECT_ID = "cbi-v15"
DATASET_ID = "features"
TABLE_ID = "technical_indicators_daily"

def calculate_indicators_incremental(symbol: str = "ZL", days_back: int = 1):
    """Calculate indicators for new data only using pandas-ta"""
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # Get new data only
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    query = f"""
    SELECT 
      date,
      symbol,
      open,
      high,
      low,
      close,
      volume
    FROM `{PROJECT_ID}.staging.market_daily`
    WHERE symbol = '{symbol}' 
      AND date >= '{start_date}'
    ORDER BY date
    """
    
    print(f"Fetching data for {symbol} from {start_date}...")
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print(f"⚠️  No new data for {symbol}")
        return
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Calculate indicators using pandas-ta
    print(f"Calculating technical indicators...")
    
    # RSI
    df.ta.rsi(length=14, append=True)
    
    # MACD
    df.ta.macd(append=True)
    
    # Bollinger Bands
    df.ta.bbands(length=20, std=2, append=True)
    
    # Moving Averages
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    
    # ATR
    df.ta.atr(length=14, append=True)
    
    # Reset index
    df.reset_index(inplace=True)
    
    # Upload to BigQuery
    print(f"Uploading {len(df)} rows to BigQuery...")
    df.to_gbq(
        f'{DATASET_ID}.{TABLE_ID}',
        project_id=PROJECT_ID,
        if_exists='append',
        table_schema=[
            {'name': 'date', 'type': 'DATE'},
            {'name': 'symbol', 'type': 'STRING'},
        ]
    )
    
    print(f"✅ Indicators calculated and uploaded for {symbol}")

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ZL"
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    calculate_indicators_incremental(symbol, days_back)

