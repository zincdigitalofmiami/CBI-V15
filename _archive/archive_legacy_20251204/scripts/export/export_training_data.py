#!/usr/bin/env python3
"""
Export training data from BigQuery to Parquet for Mac training
Exports train/val/test splits as separate files for all 5 horizons
"""
from google.cloud import bigquery
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
client = bigquery.Client(project="cbi-v15")

# Use local directory if external drive not mounted
OUTPUT_DIR = Path("TrainingData/exports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All 4 horizons (12m not available in daily_ml_matrix yet)
HORIZONS = ["1w", "1m", "3m", "6m"]

# Time-based splits (no shuffling)
SPLIT_DATES = {
    "train": ("2000-01-01", "2020-12-31"),  # ~15 years
    "val": ("2021-01-01", "2023-12-31"),    # ~3 years
    "test": ("2024-01-01", None)            # ~2 years to latest
}

def export_horizon_with_splits(horizon: str):
    """Export training table for a horizon with train/val/test splits"""
    table_ref = f"training.zl_training_minimal_{horizon}"
    target_col = f"target_{horizon}_price"
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Exporting {horizon} horizon from {table_ref}")
    logging.info(f"{'='*60}")
    
    # Query all data
    query = f"""
    SELECT * 
    FROM `cbi-v15.{table_ref}`
    WHERE {target_col} IS NOT NULL
    ORDER BY date
    """
    
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        logging.error(f"❌ Failed to query {table_ref}: {e}")
        return False
    
    if df.empty:
        logging.warning(f"⚠️  No data found in {table_ref}")
        return False
    
    logging.info(f"Total rows: {len(df):,}")
    logging.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create splits
    df['date'] = pd.to_datetime(df['date'])
    
    for split_name, (start_date, end_date) in SPLIT_DATES.items():
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) if end_date else df['date'].max()
        
        split_df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
        
        if split_df.empty:
            logging.warning(f"⚠️  No data for {split_name} split ({start_date} to {end_date or 'latest'})")
            continue
        
        # Drop date column for training (keep only features + target)
        output_cols = [col for col in split_df.columns if col not in ['date', 'symbol']]
        split_df = split_df[output_cols]
        
        output_path = OUTPUT_DIR / f"zl_training_minimal_{horizon}_{split_name}.parquet"
        split_df.to_parquet(output_path, index=False, compression='snappy')
        
        logging.info(f"✅ {split_name:5s}: {len(split_df):,} rows → {output_path.name}")
        logging.info(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return True

def export_all_horizons():
    """Export all 5 horizons with splits"""
    logging.info("🚀 Starting export of all horizons with train/val/test splits...")
    
    success_count = 0
    for horizon in HORIZONS:
        if export_horizon_with_splits(horizon):
            success_count += 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"✅ Exported {success_count}/{len(HORIZONS)} horizons successfully")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    export_all_horizons()

