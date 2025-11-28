#!/usr/bin/env python3
"""
LightGBM baseline training for ZL
Trains one model per horizon (1w, 1m, 3m, 6m)
"""
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import logging
import joblib

logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("/Volumes/Satechi Hub/Projects/CBI-V15/03_Training_Exports")
MODELS_DIR = Path("/Volumes/Satechi Hub/Projects/CBI-V15/04_Models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = {
    "target_1w_price": "1w",
    "target_1m_price": "1m",
    "target_3m_price": "3m",
    "target_6m_price": "6m"
}

def load_split(name: str) -> pd.DataFrame:
    """Load a train/val/test split from Parquet"""
    file_path = DATA_DIR / f"daily_ml_matrix_{name}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    return pd.read_parquet(file_path)

def train_lgbm_for_horizon(horizon_col: str, horizon_name: str):
    """Train LightGBM model for a specific horizon"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Training LightGBM for {horizon_name} ({horizon_col})")
    logging.info(f"{'='*60}")
    
    # Load splits
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    
    # Filter to ZL only
    train = train[train['symbol'] == 'ZL'].copy()
    val = val[val['symbol'] == 'ZL'].copy()
    test = test[test['symbol'] == 'ZL'].copy()
    
    # Filter out rows with missing target
    train = train[train[horizon_col].notna()].copy()
    val = val[val[horizon_col].notna()].copy()
    test = test[test[horizon_col].notna()].copy()
    
    logging.info(f"Train: {len(train):,} rows")
    logging.info(f"Val: {len(val):,} rows")
    logging.info(f"Test: {len(test):,} rows")
    
    # Drop non-feature columns
    meta_cols = ["date", "symbol"]
    target = horizon_col
    
    # Get all target columns to exclude
    all_targets = ["target_1w_price", "target_1m_price", "target_3m_price", "target_6m_price"]
    
    feature_cols = sorted(
        col for col in train.columns
        if col not in meta_cols + all_targets + ["price_current"]
    )
    
    logging.info(f"Features: {len(feature_cols)}")
    
    # Prepare data
    X_train = train[feature_cols].fillna(0)  # Simple null handling for now
    y_train = train[target]
    X_val = val[feature_cols].fillna(0)
    y_val = val[target]
    X_test = test[feature_cols].fillna(0)
    y_test = test[target]
    
    # Create LightGBM datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    # LightGBM parameters
    params = {
        "objective": "regression",
        "metric": ["mae", "rmse"],
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "force_row_wise": True
    }
    
    # Train model
    logging.info("Training model...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=5000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(100)
        ]
    )
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Results for {horizon_name}:")
    logging.info(f"{'='*60}")
    logging.info(f"Train - MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
    logging.info(f"Val   - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}")
    logging.info(f"Test  - MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")
    
    # Save model
    model_path = MODELS_DIR / f"lightgbm_zl_{horizon_name}.pkl"
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'horizon': horizon_name,
        'metrics': {
            'train': {'mae': mae_train, 'rmse': rmse_train, 'r2': r2_train},
            'val': {'mae': mae_val, 'rmse': rmse_val, 'r2': r2_val},
            'test': {'mae': mae_test, 'rmse': rmse_test, 'r2': r2_test}
        }
    }, model_path)
    logging.info(f"✅ Model saved to {model_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': test['date'],
        'target': y_test,
        'prediction': y_pred_test,
        'horizon': horizon_name
    })
    predictions_path = DATA_DIR / f"predictions_lightgbm_{horizon_name}.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    logging.info(f"✅ Predictions saved to {predictions_path}")
    
    return model, feature_cols

if __name__ == "__main__":
    logging.info("Starting LightGBM baseline training for ZL...")
    
    models = {}
    for horizon_col, horizon_name in HORIZONS.items():
        model, feat_cols = train_lgbm_for_horizon(horizon_col, horizon_name)
        models[horizon_name] = (model, feat_cols)
    
    logging.info("\n✅ All models trained successfully!")
    logging.info(f"Models saved to: {MODELS_DIR}")

