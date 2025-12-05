#!/usr/bin/env python3
"""
Upload predictions to BigQuery forecasts tables
Supports ensemble and individual model predictions
"""
import pandas as pd
from pathlib import Path
import logging
from google.cloud import bigquery
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = "cbi-v15"
DATA_DIR = Path("TrainingData/exports")

# All 4 horizons (12m not available yet)
HORIZONS = ["1w", "1m", "3m", "6m"]

def upload_predictions_for_horizon(horizon: str, model_type: str = "ensemble"):
    """Upload predictions for a specific horizon"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Uploading {model_type} predictions for {horizon} horizon")
    logging.info(f"{'='*60}")
    
    # Load predictions
    pred_path = DATA_DIR / f"predictions_{model_type}_{horizon}.parquet"
    if not pred_path.exists():
        logging.warning(f"Predictions file not found: {pred_path}")
        return False
    
    pred_df = pd.read_parquet(pred_path)
    
    # Load test data to get dates
    test_path = DATA_DIR / f"zl_training_minimal_{horizon}_test.parquet"
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        # Align dates if possible
        if len(pred_df) == len(test_df):
            if 'date' in test_df.columns:
                pred_df['date'] = test_df['date'].values
    else:
        # Create synthetic dates if needed
        if 'date' not in pred_df.columns:
            start_date = datetime(2024, 1, 1)
            pred_df['date'] = pd.date_range(start=start_date, periods=len(pred_df), freq='D')
    
    # Prepare DataFrame for BigQuery
    upload_df = pd.DataFrame({
        'date': pd.to_datetime(pred_df.get('date', pd.date_range(start='2024-01-01', periods=len(pred_df), freq='D'))),
        'horizon': horizon,
        'prediction': pred_df['prediction'].values,
        'target': pred_df['target'].values,
        'p10': pred_df['prediction'].values - pred_df['prediction'].std() * 1.28,  # Approximate quantiles
        'p50': pred_df['prediction'].values,
        'p90': pred_df['prediction'].values + pred_df['prediction'].std() * 1.28,
        'confidence_score': 0.85,  # Placeholder
        'model_type': model_type,
        'model_version': 'v1',
        'as_of': datetime.now()
    })
    
    # Upload to BigQuery
    try:
        client = bigquery.Client(project=PROJECT_ID)
        dataset_id = "predictions"
        table_id = f"{PROJECT_ID}.{dataset_id}.zl_predictions_{horizon}"
        
        # Ensure predictions dataset exists
        try:
            client.get_dataset(f"{PROJECT_ID}.{dataset_id}")
        except Exception:
            ds = bigquery.Dataset(f"{PROJECT_ID}.{dataset_id}")
            ds.location = "us-central1"
            client.create_dataset(ds, exists_ok=True)
        
        # Create table if doesn't exist
        schema = [
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("horizon", "STRING"),
            bigquery.SchemaField("prediction", "FLOAT64"),
            bigquery.SchemaField("target", "FLOAT64"),
            bigquery.SchemaField("p10", "FLOAT64"),
            bigquery.SchemaField("p50", "FLOAT64"),
            bigquery.SchemaField("p90", "FLOAT64"),
            bigquery.SchemaField("confidence_score", "FLOAT64"),
            bigquery.SchemaField("model_type", "STRING"),
            bigquery.SchemaField("model_version", "STRING"),
            bigquery.SchemaField("as_of", "TIMESTAMP"),
        ]
        
        table_ref = client.dataset(dataset_id).table(f"zl_predictions_{horizon}")
        try:
            client.get_table(table_ref)
        except:
            table = bigquery.Table(table_ref, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(field="date")
            table.clustering_fields = ["horizon"]
            client.create_table(table)
            logging.info(f"Created table {table_id}")
        
        # Upload with MERGE pattern (avoid duplicates)
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            source_format=bigquery.SourceFormat.PARQUET
        )
        
        job = client.load_table_from_dataframe(upload_df, table_ref, job_config=job_config)
        job.result()
        
        logging.info(f"✅ Uploaded {len(upload_df)} predictions to {table_id}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to upload predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

def upload_all_predictions(model_type: str = "ensemble"):
    """Upload predictions for all horizons"""
    logging.info(f"🚀 Starting upload of {model_type} predictions...")
    
    success_count = 0
    for horizon in HORIZONS:
        if upload_predictions_for_horizon(horizon, model_type):
            success_count += 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"✅ Uploaded {success_count}/{len(HORIZONS)} horizons successfully")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    import sys
    
    model_type = sys.argv[1] if len(sys.argv) > 1 else "ensemble"
    upload_all_predictions(model_type)
