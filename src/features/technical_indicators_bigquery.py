#!/usr/bin/env python3
"""
Technical Indicators Calculation - BigQuery SQL Approach
Optimized for 15-year initial load
"""

from google.cloud import bigquery
from pathlib import Path

PROJECT_ID = "cbi-v15"
DATASET_ID = "features"
TABLE_ID = "technical_indicators_15y"

def create_technical_indicators_table():
    """Create technical indicators table using BigQuery SQL"""
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # Load UDFs
    udf_sql = Path(__file__).parent.parent.parent / "dataform" / "includes" / "technical_indicators_udf.sqlx"
    
    # Main query to calculate all indicators
    query = f"""
    -- Load UDFs
    {udf_sql.read_text() if udf_sql.exists() else ''}
    
    -- Calculate all technical indicators for 15 years
    SELECT 
      date,
      symbol,
      close,
      
      -- RSI
      calculate_rsi(
        ARRAY_AGG(close) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ),
        14
      ) AS rsi_14,
      
      -- Moving Averages
      calculate_sma(
        ARRAY_AGG(close) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ),
        10
      ) AS sma_10,
      
      calculate_sma(
        ARRAY_AGG(close) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ),
        20
      ) AS sma_20,
      
      calculate_sma(
        ARRAY_AGG(close) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ),
        50
      ) AS sma_50,
      
      calculate_sma(
        ARRAY_AGG(close) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
        ),
        200
      ) AS sma_200,
      
      -- Bollinger Bands
      calculate_bollinger(
        ARRAY_AGG(close) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ),
        20, 2.0
      ) AS bollinger_bands,
      
      -- ATR
      calculate_atr(
        ARRAY_AGG(high) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ),
        ARRAY_AGG(low) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ),
        ARRAY_AGG(close) OVER (
          PARTITION BY symbol 
          ORDER BY date 
          ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ),
        14
      ) AS atr_14
      
    FROM `{PROJECT_ID}.staging.market_daily`
    WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 15 YEAR)
      AND symbol IN ('ZL', 'ZS', 'ZM', 'CL', 'HO', 'FCPO')
    ORDER BY symbol, date
    """
    
    # Run query and save results
    job_config = bigquery.QueryJobConfig(
        destination=f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}",
        write_disposition="WRITE_TRUNCATE"  # Overwrite for initial load
    )
    
    print(f"Calculating technical indicators for 15 years...")
    job = client.query(query, job_config=job_config)
    job.result()  # Wait for completion
    
    print(f"✅ Technical indicators calculated!")
    print(f"   Rows processed: {job.num_dml_affected_rows:,}")
    print(f"   Bytes processed: {job.total_bytes_processed / 1024 / 1024 / 1024:.2f} GB")
    print(f"   Cost: ${job.total_bytes_processed / 1024 / 1024 / 1024 / 1024 * 5:.2f}")

if __name__ == "__main__":
    create_technical_indicators_table()

