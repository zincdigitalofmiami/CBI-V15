#!/usr/bin/env python3
"""
Export training data from BigQuery to Parquet for Mac training
Exports train/val/test splits as separate files
"""
from google.cloud import bigquery
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
client = bigquery.Client(project="cbi-v15")

OUTPUT_DIR = Path("/Volumes/Satechi Hub/Projects/CBI-V15/03_Training_Exports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": "training.daily_ml_matrix_train",
    "val": "training.daily_ml_matrix_val",
    "test": "training.daily_ml_matrix_test"
}

def export_split(split_name: str, table_ref: str):
    """Export a single split to Parquet"""
    logging.info(f"Exporting {split_name} split from {table_ref}...")
    
    query = f"SELECT * FROM `cbi-v15.{table_ref}` ORDER BY date, symbol"
    df = client.query(query).to_dataframe()
    
    if df.empty:
        logging.warning(f"⚠️  No data found for {split_name} split")
        return False
    
    output_path = OUTPUT_DIR / f"daily_ml_matrix_{split_name}.parquet"
    df.to_parquet(output_path, index=False, compression='snappy')
    
    logging.info(f"✅ Exported {len(df):,} rows, {len(df.columns)} columns to {output_path}")
    logging.info(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return True

def export_all_splits():
    """Export all train/val/test splits"""
    logging.info("Exporting training data splits from BigQuery...")
    
    for split_name, table_ref in SPLITS.items():
        export_split(split_name, table_ref)
    
    logging.info("✅ All splits exported successfully")

if __name__ == "__main__":
    export_all_splits()

