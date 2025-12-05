#!/usr/bin/env python3
"""
LightGBM baseline training for ZL - Minimal Feature Set

- Trains one model per horizon (1w, 1m, 3m, 6m)
- Uses returns as the target:
    y = (target_price - price_current) / price_current
- Logs MAE / RMSE / R² in price space and MAPE (% error) for baselines.
"""
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import logging
import joblib

from src.training.utils.metrics import sharpe_ratio, sortino_ratio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use local directory structure
DATA_DIR = Path("TrainingData/exports")
MODELS_DIR = Path("Models/local/baseline")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# All 4 horizons (12m not available yet)
HORIZONS = ["1w", "1m", "3m", "6m"]

def load_split(horizon: str, split_name: str) -> pd.DataFrame:
    """Load a train/val/test split from Parquet for specific horizon"""
    file_path = DATA_DIR / f"zl_training_minimal_{horizon}_{split_name}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    return pd.read_parquet(file_path)

def train_lgbm_for_horizon(horizon: str):
    """Train LightGBM model for a specific horizon"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Training LightGBM for {horizon} horizon")
    logging.info(f"{'='*60}")
    
    # Load splits
    train = load_split(horizon, "train")
    val = load_split(horizon, "val")
    test = load_split(horizon, "test")
    
    # Target column name (price level in the parquet)
    target_col = f"target_{horizon}_price"
    
    # Filter out rows with missing or non‑positive targets
    train = train[train[target_col].notna()].copy()
    val = val[val[target_col].notna()].copy()
    test = test[test[target_col].notna()].copy()
    
    logging.info(f"Train: {len(train):,} rows")
    logging.info(f"Val: {len(val):,} rows")
    logging.info(f"Test: {len(test):,} rows")
    
    # Price column used for return targets
    price_col = "price_current" if "price_current" in train.columns else "price"
    if price_col not in train.columns:
        raise KeyError(f"Expected price column 'price_current' or 'price' not found in training data for {horizon}")
    
    # Feature columns: all except target(s), price_current, and non-numeric identifiers
    feature_cols = sorted(
        col for col in train.columns
        if col not in [
            target_col,
            "price_current",
            "date",
            "symbol",
            "regime",
        ] and not col.startswith("target_")
    )
    
    logging.info(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    
    # Prepare feature matrices
    def _coerce_features(df_in: pd.DataFrame) -> pd.DataFrame:
        df_num = df_in.copy()
        for c in df_num.columns:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
        return df_num.fillna(0.0)

    X_train = _coerce_features(train[feature_cols])
    X_val = _coerce_features(val[feature_cols])
    X_test = _coerce_features(test[feature_cols])

    # Price levels and targets (price space)
    train_price = train[price_col].astype(float).values
    val_price = val[price_col].astype(float).values
    test_price = test[price_col].astype(float).values

    train_target_price = train[target_col].astype(float).values
    val_target_price = val[target_col].astype(float).values
    test_target_price = test[target_col].astype(float).values

    # Drop any rows with non‑positive or NaN prices before computing returns
    def _clean(price_arr, target_arr, X):
        mask = (
            np.isfinite(price_arr)
            & np.isfinite(target_arr)
            & (price_arr > 0.0)
            & (target_arr > 0.0)
        )
        return price_arr[mask], target_arr[mask], X[mask]

    train_price, train_target_price, X_train = _clean(train_price, train_target_price, X_train.values)
    val_price, val_target_price, X_val = _clean(val_price, val_target_price, X_val.values)
    test_price, test_target_price, X_test = _clean(test_price, test_target_price, X_test.values)

    # Convert back to DataFrames after masking
    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_val = pd.DataFrame(X_val, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    # Targets in return space (relative move vs current price)
    y_train = (train_target_price - train_price) / train_price
    y_val = (val_target_price - val_price) / val_price
    y_test = (test_target_price - test_price) / test_price
    
    # Create LightGBM datasets (use numpy arrays to avoid pandas dtype issues)
    train_set = lgb.Dataset(X_train.values, label=y_train, feature_name=feature_cols)
    val_set = lgb.Dataset(X_val.values, label=y_val, reference=train_set)
    
    # LightGBM parameters (tuned ranges per plan)
    params = {
        "objective": "regression",
        "metric": ["mae", "rmse"],
        "learning_rate": 0.03,  # Will tune 0.01-0.05
        "num_leaves": 63,  # Will tune 31-127
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,  # Will tune 50-100
        "max_depth": -1,  # Unlimited
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
    
    # Evaluate in return space
    y_pred_train = model.predict(X_train.values)
    y_pred_val = model.predict(X_val.values)
    y_pred_test = model.predict(X_test.values)
    
    mae_train_ret = mean_absolute_error(y_train, y_pred_train)
    mae_val_ret = mean_absolute_error(y_val, y_pred_val)
    mae_test_ret = mean_absolute_error(y_test, y_pred_test)
    
    rmse_train_ret = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val_ret = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test_ret = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    r2_train_ret = r2_score(y_train, y_pred_train)
    r2_val_ret = r2_score(y_val, y_pred_val)
    r2_test_ret = r2_score(y_test, y_pred_test)

    # Convert predictions back to price space for MAE / MAPE in dollars/percent
    train_pred_price = train_price * (1.0 + y_pred_train)
    val_pred_price = val_price * (1.0 + y_pred_val)
    test_pred_price = test_price * (1.0 + y_pred_test)

    mae_train = mean_absolute_error(train_target_price, train_pred_price)
    mae_val = mean_absolute_error(val_target_price, val_pred_price)
    mae_test = mean_absolute_error(test_target_price, test_pred_price)
    
    rmse_train = np.sqrt(mean_squared_error(train_target_price, train_pred_price))
    rmse_val = np.sqrt(mean_squared_error(val_target_price, val_pred_price))
    rmse_test = np.sqrt(mean_squared_error(test_target_price, test_pred_price))
    
    r2_train = r2_score(train_target_price, train_pred_price)
    r2_val = r2_score(val_target_price, val_pred_price)
    r2_test = r2_score(test_target_price, test_pred_price)

    # MAPE in percent (avoid division by zero)
    eps = 1e-8
    mape_train = np.mean(np.abs((train_target_price - train_pred_price) / (np.abs(train_target_price) + eps))) * 100.0
    mape_val = np.mean(np.abs((val_target_price - val_pred_price) / (np.abs(val_target_price) + eps))) * 100.0
    mape_test = np.mean(np.abs((test_target_price - test_pred_price) / (np.abs(test_target_price) + eps))) * 100.0
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Results for {horizon}:")
    logging.info(f"{'='*60}")
    logging.info(f"Train - MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}, MAPE: {mape_train:.2f}%")
    logging.info(f"Val   - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}, MAPE: {mape_val:.2f}%")
    logging.info(f"Test  - MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}, MAPE: {mape_test:.2f}%")

    # ------------------------------------------------------------------
    # Risk-style metrics: Sharpe / Sortino for a simple sign-based strategy
    # Long if predicted return > 0, short if predicted return < 0.
    # Strategy return = sign(pred) * realized return.
    # ------------------------------------------------------------------
    strat_ret_test = np.sign(y_pred_test) * y_test
    sharpe_test = sharpe_ratio(strat_ret_test, risk_free_rate=0.0, periods_per_year=252)
    sortino_test = sortino_ratio(strat_ret_test, risk_free_rate=0.0, periods_per_year=252)
    logging.info(f"Test  - Sharpe: {sharpe_test:.3f}, Sortino: {sortino_test:.3f}")
    
    # Success criteria check (MAPE-based)
    if mape_test < 4.0:
        logging.info(f"✅ Success criteria met: Test MAPE < 4% (MAPE={mape_test:.2f}%)")
    else:
        logging.warning(f"⚠️  Success criteria not met: MAPE={mape_test:.2f}% (target <4%)")
    
    # Overfitting check
    if abs(mae_val - mae_train) / (mae_train + 1e-8) > 0.10:
        logging.warning(f"⚠️  Possible overfitting: val MAE ({mae_val:.4f}) >10% different from train MAE ({mae_train:.4f})")
    
    # Save model
    model_path = MODELS_DIR / f"zl_{horizon}_lightgbm.pkl"
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'horizon': horizon,
        'target_col': target_col,
        'metrics': {
            'train': {
                'mae': mae_train,
                'rmse': rmse_train,
                'r2': r2_train,
                'mape': mape_train,
            },
            'val': {
                'mae': mae_val,
                'rmse': rmse_val,
                'r2': r2_val,
                'mape': mape_val,
            },
            'test': {
                'mae': mae_test,
                'rmse': rmse_test,
                'r2': r2_test,
                'mape': mape_test,
                'sharpe': sharpe_test,
                'sortino': sortino_test,
            },
        },
        'hyperparameters': params
    }, model_path)
    logging.info(f"✅ Model saved to {model_path}")
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    importance_path = MODELS_DIR / f"zl_{horizon}_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logging.info(f"✅ Feature importance saved to {importance_path}")
    logging.info(f"   Top 5 features: {', '.join(importance_df.head(5)['feature'].tolist())}")
    
    # Save predictions (price space)
    predictions_df = pd.DataFrame({
        'target': test_target_price,
        'prediction': test_pred_price,
        'horizon': horizon
    })
    predictions_path = DATA_DIR / f"predictions_baseline_{horizon}.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    logging.info(f"✅ Predictions saved to {predictions_path}")
    
    return model, feature_cols, {
        'mae_test': mae_test,
        'r2_test': r2_test,
        'mae_val': mae_val
    }

if __name__ == "__main__":
    logging.info("🚀 Starting LightGBM baseline training for ZL (minimal feature set)...")
    
    models = {}
    results = {}
    failed = []
    for horizon in HORIZONS:
        try:
            model, feat_cols, metrics = train_lgbm_for_horizon(horizon)
            models[horizon] = (model, feat_cols)
            results[horizon] = metrics
        except Exception as e:
            logging.error(f"❌ Failed to train {horizon}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(horizon)
    
    logging.info(f"\n{'='*60}")
    if failed:
        logging.info("⚠️ Training loop completed with failures:")
        logging.info(f"   Failed horizons: {', '.join(failed)}")
    else:
        logging.info("✅ All models trained successfully!")
    logging.info(f"Models saved to: {MODELS_DIR}")
    logging.info(f"\nSummary:")
    for horizon, metrics in results.items():
        logging.info(f"  {horizon:3s}: MAE={metrics['mae_test']:.4f}, R²={metrics['r2_test']:.4f}")
    logging.info(f"{'='*60}")
