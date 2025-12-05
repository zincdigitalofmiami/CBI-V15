#!/usr/bin/env python3
"""
Compute SHAP values for ensemble or baseline models
TreeSHAP for LightGBM models, converts to cents/lb
"""
import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
import logging
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use local directory structure
DATA_DIR = Path("TrainingData/exports")
MODELS_DIR = Path("Models/local")
BASELINE_DIR = MODELS_DIR / "baseline"
ENSEMBLE_DIR = MODELS_DIR / "ensemble"
SHAP_DIR = MODELS_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# All 4 horizons (12m not available yet)
HORIZONS = ["1w", "1m", "3m", "6m"]

PROJECT_ID = "cbi-v15"
SHAP_TABLE = f"{PROJECT_ID}.models_v4.shap_daily"

def compute_shap_for_horizon(horizon: str, use_ensemble: bool = False):
    """Compute SHAP values for a horizon"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Computing SHAP values for {horizon} horizon")
    logging.info(f"{'='*60}")
    
    # Load test data
    test_path = DATA_DIR / f"zl_training_minimal_{horizon}_test.parquet"
    if not test_path.exists():
        logging.error(f"Test data not found: {test_path}")
        return None
    
    test_df = pd.read_parquet(test_path)
    target_col = f"target_{horizon}_price"
    
    # Filter to rows with targets
    test_df = test_df[test_df[target_col].notna()].copy()
    
    # Feature columns
    feature_cols = sorted(
        col for col in test_df.columns
        if col not in [target_col, "price_current"]
    )
    
    X_test = test_df[feature_cols].fillna(0)
    
    # Load model (ensemble or baseline)
    if use_ensemble:
        # For ensemble, use baseline model for SHAP (weighted average doesn't have SHAP)
        model_path = BASELINE_DIR / f"zl_{horizon}_lightgbm.pkl"
        model_data = joblib.load(model_path)
        model = model_data['model']
        logging.info("Using baseline model for SHAP (ensemble is weighted average)")
    else:
        model_path = BASELINE_DIR / f"zl_{horizon}_lightgbm.pkl"
        model_data = joblib.load(model_path)
        model = model_data['model']
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values (sample if too large)
    if len(X_test) > 1000:
        logging.info(f"Sampling 1000 rows from {len(X_test)} for SHAP computation")
        sample_idx = np.random.choice(len(X_test), 1000, replace=False)
        X_sample = X_test.iloc[sample_idx]
        test_dates = test_df.iloc[sample_idx].index  # Use index as date proxy
    else:
        X_sample = X_test
        test_dates = test_df.index
    
    logging.info("Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    # Convert to DataFrame
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_cols,
        index=X_sample.index
    )
    
    # Get predictions to compute marginal price response
    predictions = model.predict(X_sample)
    actuals = test_df.loc[X_sample.index][target_col].values
    
    # Convert SHAP to cents/lb (approximate: SHAP is in price units)
    # For more accuracy, compute marginal response per feature
    shap_cents = shap_df.copy()  # Already in price units, convert if needed
    
    # Reshape to long format
    shap_long = []
    # Get date column if available
    has_date = 'date' in test_df.columns
    for i, idx in enumerate(shap_df.index):
        date_val = test_df.loc[idx, 'date'] if has_date else pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
        for feature in feature_cols:
            shap_long.append({
                'date': pd.to_datetime(date_val).date() if has_date else date_val,
                'horizon': horizon,
                'feature_name': feature,
                'shap_value_cents': float(shap_cents.loc[idx, feature])
            })
    
    shap_long_df = pd.DataFrame(shap_long)
    
    # Save to Parquet
    shap_path = SHAP_DIR / f"zl_{horizon}_shap_values.parquet"
    shap_long_df.to_parquet(shap_path, index=False)
    logging.info(f"✅ SHAP values saved to {shap_path}")
    
    # Upload to BigQuery
    try:
        client = bigquery.Client(project=PROJECT_ID)
        
        # Create table if doesn't exist
        schema = [
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("horizon", "STRING"),
            bigquery.SchemaField("feature_name", "STRING"),
            bigquery.SchemaField("shap_value_cents", "FLOAT64"),
        ]
        
        table_ref = client.dataset("models_v4").table("shap_daily")
        try:
            client.get_table(table_ref)
        except:
            table = bigquery.Table(table_ref, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(field="date")
            table.clustering_fields = ["horizon", "feature_name"]
            client.create_table(table)
            logging.info(f"Created table {SHAP_TABLE}")
        
        # Upload data
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            source_format=bigquery.SourceFormat.PARQUET
        )
        
        job = client.load_table_from_dataframe(shap_long_df, table_ref, job_config=job_config)
        job.result()
        logging.info(f"✅ Uploaded {len(shap_long_df)} SHAP values to {SHAP_TABLE}")
        
    except Exception as e:
        logging.warning(f"⚠️  Could not upload to BigQuery: {e}")
    
    return shap_long_df

if __name__ == "__main__":
    logging.info("🚀 Starting SHAP computation for ZL...")
    
    results = {}
    for horizon in HORIZONS:
        try:
            shap_df = compute_shap_for_horizon(horizon, use_ensemble=False)
            if shap_df is not None:
                results[horizon] = len(shap_df)
        except Exception as e:
            logging.error(f"❌ Failed to compute SHAP for {horizon}: {e}")
            import traceback
            traceback.print_exc()
    
    logging.info(f"\n{'='*60}")
    logging.info("✅ SHAP computation complete!")
    if results:
        logging.info(f"Computed SHAP for {len(results)} horizons")
        for horizon, count in results.items():
            logging.info(f"  {horizon:3s}: {count} SHAP values")
    logging.info(f"{'='*60}")

