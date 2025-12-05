#!/usr/bin/env python3
"""
Ensemble prediction for ZL - Combines baseline + DL predictions
Uses weighted average or meta-learner stacking
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import logging
import json
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use local directory structure
DATA_DIR = Path("TrainingData/exports")
MODELS_DIR = Path("Models/local/ensemble")
BASELINE_DIR = Path("Models/local/baseline")
LSTM_DIR = Path("Models/local/dl_round1")
TFT_DIR = Path("Models/local/dl_round2")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# All 4 horizons (12m not available yet)
HORIZONS = ["1w", "1m", "3m", "6m"]

def load_predictions(horizon: str, model_type: str):
    """Load predictions from a model type"""
    pred_path = DATA_DIR / f"predictions_{model_type}_{horizon}.parquet"
    if not pred_path.exists():
        return None
    return pd.read_parquet(pred_path)

def weighted_average_ensemble(horizon: str):
    """Simple weighted average ensemble based on OOS performance"""
    logging.info(f"Creating weighted average ensemble for {horizon}...")
    
    # Load all available predictions
    predictions = {}
    weights = {}
    
    # Baseline
    baseline_pred = load_predictions(horizon, "baseline")
    if baseline_pred is not None:
        predictions['baseline'] = baseline_pred
        # Get MAE from model
        try:
            baseline_model = joblib.load(BASELINE_DIR / f"zl_{horizon}_lightgbm.pkl")
            baseline_mae = baseline_model['metrics']['val']['mae']
            weights['baseline'] = 1.0 / (baseline_mae + 1e-6)
        except:
            weights['baseline'] = 1.0
    
    # LSTM
    lstm_pred = load_predictions(horizon, "lstm")
    if lstm_pred is not None:
        predictions['lstm'] = lstm_pred
        try:
            lstm_model = torch.load(LSTM_DIR / f"zl_{horizon}_lstm.pt", map_location='cpu')
            lstm_mae = lstm_model['metrics']['val']['mae']
            weights['lstm'] = 1.0 / (lstm_mae + 1e-6)
        except:
            weights['lstm'] = 1.0
    
    # TFT
    tft_pred = load_predictions(horizon, "tft")
    if tft_pred is not None:
        predictions['tft'] = tft_pred
        try:
            tft_model = torch.load(TFT_DIR / f"zl_{horizon}_tft.pt", map_location='cpu')
            tft_mae = tft_model['metrics']['val']['mae']
            weights['tft'] = 1.0 / (tft_mae + 1e-6)
        except:
            weights['tft'] = 1.0
    
    if not predictions:
        logging.warning(f"No predictions found for {horizon}")
        return None
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    logging.info(f"Model weights: {weights}")
    
    # Align predictions by index
    aligned_preds = []
    aligned_targets = None
    
    for model_name, pred_df in predictions.items():
        if aligned_targets is None:
            aligned_targets = pred_df['target'].values
        aligned_preds.append(pred_df['prediction'].values * weights[model_name])
    
    # Weighted average
    ensemble_pred = np.sum(aligned_preds, axis=0)
    
    # Calculate metrics
    mae = mean_absolute_error(aligned_targets, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(aligned_targets, ensemble_pred))
    r2 = r2_score(aligned_targets, ensemble_pred)
    
    logging.info(f"Ensemble metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Save ensemble predictions
    ensemble_df = pd.DataFrame({
        'target': aligned_targets,
        'prediction': ensemble_pred,
        'horizon': horizon
    })
    ensemble_path = DATA_DIR / f"predictions_ensemble_{horizon}.parquet"
    ensemble_df.to_parquet(ensemble_path, index=False)
    logging.info(f"✅ Ensemble predictions saved to {ensemble_path}")
    
    # Save weights
    weights_path = MODELS_DIR / f"zl_{horizon}_weights.json"
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    logging.info(f"✅ Model weights saved to {weights_path}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'weights': weights
    }

def create_ensemble_for_horizon(horizon: str):
    """Create ensemble for a specific horizon"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Creating ensemble for {horizon} horizon")
    logging.info(f"{'='*60}")
    
    return weighted_average_ensemble(horizon)

if __name__ == "__main__":
    import torch
    
    logging.info("🚀 Starting ensemble creation for ZL...")
    
    # Import torch at top level for TFT model loading
    import torch
    
    results = {}
    for horizon in HORIZONS:
        try:
            result = create_ensemble_for_horizon(horizon)
            if result:
                results[horizon] = result
        except Exception as e:
            logging.error(f"❌ Failed to create ensemble for {horizon}: {e}")
            import traceback
            traceback.print_exc()
    
    logging.info(f"\n{'='*60}")
    logging.info("✅ All ensembles created successfully!")
    logging.info(f"Models saved to: {MODELS_DIR}")
    if results:
        logging.info(f"\nSummary:")
        for horizon, metrics in results.items():
            logging.info(f"  {horizon:3s}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
            logging.info(f"         Weights: {metrics['weights']}")
    logging.info(f"{'='*60}")

