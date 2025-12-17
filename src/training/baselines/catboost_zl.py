#!/usr/bin/env python3
"""
CatBoost Quantile Regression for ZL
Trains quantile models (P10, P50, P90) per horizon for probabilistic forecasts
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use relative paths from project root
ROOT_DIR = Path(__file__).resolve().parents[3]  # Go up from src/training/baselines/
MODELS_DIR = ROOT_DIR / "models" / "baselines" / "catboost"

# Create directories if they don't exist (on first run)
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    # Running in read-only mode or sandbox - directories will be created on actual training
    pass

HORIZONS = {
    "target_price_1w": "1w",
    "target_price_1m": "1m",
    "target_price_3m": "3m",
    "target_price_6m": "6m",
    # Note: 12m not available in current macro output
}

QUANTILES = [0.1, 0.5, 0.9]  # P10, P50, P90


def train_catboost_quantile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    alpha: float = 0.5,
    horizon_name: str = "1w",
) -> CatBoostRegressor:
    """
    Train a single CatBoost quantile regression model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        alpha: Quantile to predict (0.5 = median)
        horizon_name: Horizon identifier

    Returns:
        Trained CatBoost model
    """
    logger.info(f"Training CatBoost for P{int(alpha*100)} quantile...")

    # CatBoost parameters for quantile regression
    params = {
        "loss_function": f"Quantile:alpha={alpha}",
        "iterations": 5000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": 100,
        "early_stopping_rounds": 200,
        "task_type": "CPU",  # Use "GPU" if available
    }

    model = CatBoostRegressor(**params)

    # Create CatBoost pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # Train
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False,
    )

    return model


def evaluate_quantile_model(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
) -> Dict[str, float]:
    """
    Evaluate a quantile regression model using pinball loss.

    Args:
        model: Trained CatBoost model
        X, y: Test data
        alpha: Quantile level

    Returns:
        Dict with metrics (pinball_loss, mae, coverage)
    """
    y_pred = model.predict(X)

    # Pinball loss for quantile evaluation
    errors = y - y_pred
    pinball = np.where(
        errors >= 0,
        alpha * errors,
        (alpha - 1) * errors,
    )
    pinball_loss = np.mean(pinball)

    # Coverage (for P10 and P90)
    if alpha == 0.1:
        coverage = np.mean(y >= y_pred)  # Should be ~90%
    elif alpha == 0.9:
        coverage = np.mean(y <= y_pred)  # Should be ~90%
    else:
        coverage = np.nan

    # Also compute MAE for P50 (median)
    mae = mean_absolute_error(y, y_pred)

    return {
        "pinball_loss": float(pinball_loss),
        "mae": float(mae),
        "coverage": float(coverage) if not np.isnan(coverage) else None,
    }


def train_catboost_for_horizon(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_col: str,
    horizon_name: str,
) -> Tuple[Dict[float, CatBoostRegressor], Dict[str, Any]]:
    """
    Train CatBoost quantile models for a specific horizon.

    Trains 3 models: P10, P50, P90 for probabilistic forecasting.

    Args:
        train_df, val_df, test_df: Data splits
        horizon_col: Target column name
        horizon_name: Horizon identifier (e.g., '1w')

    Returns:
        (models dict, metrics dict)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training CatBoost Quantile Models for {horizon_name}")
    logger.info(f"{'='*60}")

    # Filter to ZL only
    train = train_df[train_df["symbol"] == "ZL"].copy()
    val = val_df[val_df["symbol"] == "ZL"].copy()
    test = test_df[test_df["symbol"] == "ZL"].copy()

    # Filter out missing targets
    train = train[train[horizon_col].notna()].copy()
    val = val[val[horizon_col].notna()].copy()
    test = test[test[horizon_col].notna()].copy()

    logger.info(f"Train: {len(train):,} rows")
    logger.info(f"Val: {len(val):,} rows")
    logger.info(f"Test: {len(test):,} rows")

    # Feature columns
    meta_cols = [
        "as_of_date",
        "symbol",
        "train_val_test_split",
        "updated_at",
        "training_weight",
    ]
    all_targets = list(HORIZONS.keys())
    feature_cols = sorted(
        col
        for col in train.columns
        if col not in meta_cols + all_targets + ["price_current"]
    )

    logger.info(f"Features: {len(feature_cols)}")

    # Prepare data
    X_train = train[feature_cols].fillna(0)
    y_train = train[horizon_col]
    X_val = val[feature_cols].fillna(0)
    y_val = val[horizon_col]
    X_test = test[feature_cols].fillna(0)
    y_test = test[horizon_col]

    # Train one model per quantile
    models = {}
    all_metrics = {}

    for alpha in QUANTILES:
        quantile_name = f"P{int(alpha*100)}"
        logger.info(f"\n--- Training {quantile_name} ---")

        model = train_catboost_quantile(
            X_train, y_train, X_val, y_val, alpha=alpha, horizon_name=horizon_name
        )
        models[alpha] = model

        # Evaluate on test set
        metrics = evaluate_quantile_model(model, X_test, y_test, alpha)
        all_metrics[quantile_name] = metrics

        logger.info(f"  {quantile_name} Metrics:")
        logger.info(f"    Pinball Loss: {metrics['pinball_loss']:.4f}")
        logger.info(f"    MAE: {metrics['mae']:.4f}")
        if metrics["coverage"] is not None:
            logger.info(f"    Coverage: {metrics['coverage']:.2%}")

        # Save model
        model_path = MODELS_DIR / f"catboost_zl_{horizon_name}_{quantile_name}.cbm"
        model.save_model(str(model_path))
        logger.info(f"    ✅ Saved to {model_path}")

    # Save combined artifact with all quantiles
    combined_path = MODELS_DIR / f"catboost_zl_{horizon_name}_all_quantiles.pkl"
    joblib.dump(
        {
            "models": {f"P{int(a*100)}": m for a, m in models.items()},
            "feature_cols": feature_cols,
            "horizon": horizon_name,
            "metrics": all_metrics,
        },
        combined_path,
    )
    logger.info(f"\n✅ All quantiles saved to {combined_path}")

    return models, all_metrics


def load_training_data_from_motherduck(
    motherduck_token: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training data from MotherDuck training.daily_ml_matrix_zl.

    Splits data into train/val/test using time-based splits:
    - Train: 2018-01-01 to 2022-12-31
    - Val: 2023-01-01 to 2023-06-30
    - Test: 2023-07-01 to present

    Args:
        motherduck_token: Optional MotherDuck token (uses env if not provided)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    import os
    import duckdb

    token = motherduck_token or os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise ValueError("MOTHERDUCK_TOKEN not set")

    db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    conn = duckdb.connect(f"md:{db_name}?motherduck_token={token}")

    logger.info("Loading training data from MotherDuck...")

    # Load full dataset
    query = """
    SELECT *
    FROM training.daily_ml_matrix_zl
    WHERE symbol = 'ZL'
    ORDER BY as_of_date
    """

    try:
        df = conn.execute(query).df()
    except Exception as e:
        logger.error(f"Failed to load from MotherDuck: {e}")
        conn.close()
        raise

    conn.close()

    if df.empty:
        raise ValueError("No data returned from training.daily_ml_matrix_zl")

    logger.info(f"Loaded {len(df):,} rows from MotherDuck")

    # Time-based splits
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    train_df = df[df["as_of_date"] < "2023-01-01"].copy()
    val_df = df[
        (df["as_of_date"] >= "2023-01-01") & (df["as_of_date"] < "2023-07-01")
    ].copy()
    test_df = df[df["as_of_date"] >= "2023-07-01"].copy()

    logger.info(f"Train: {len(train_df):,} rows (< 2023-01-01)")
    logger.info(f"Val: {len(val_df):,} rows (2023-01-01 to 2023-06-30)")
    logger.info(f"Test: {len(test_df):,} rows (>= 2023-07-01)")

    return train_df, val_df, test_df


def main():
    """Train CatBoost quantile models for all horizons."""
    logger.info("=" * 70)
    logger.info("CatBoost Quantile Training for ZL")
    logger.info("=" * 70)

    # Load training data from MotherDuck
    try:
        train_df, val_df, test_df = load_training_data_from_motherduck()
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        logger.info("\nTo generate training data, run:")
        logger.info("  python src/engines/anofox/build_training.py")
        return

    # Train for each horizon
    all_results = {}

    for horizon_col, horizon_name in HORIZONS.items():
        if horizon_col not in train_df.columns:
            logger.warning(
                f"Target column {horizon_col} not found, skipping {horizon_name}"
            )
            continue

        try:
            models, metrics = train_catboost_for_horizon(
                train_df, val_df, test_df, horizon_col, horizon_name
            )
            all_results[horizon_name] = {"models": models, "metrics": metrics}
        except Exception as e:
            logger.error(f"Failed to train {horizon_name}: {e}")
            continue

    logger.info("\n" + "=" * 70)
    logger.info("✅ CatBoost Training Complete")
    logger.info(f"   Horizons trained: {list(all_results.keys())}")
    logger.info(f"   Models saved to: {MODELS_DIR}")
    logger.info("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
