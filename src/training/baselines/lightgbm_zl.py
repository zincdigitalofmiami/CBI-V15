#!/usr/bin/env python3
"""
LightGBM baseline training for ZL
Trains one model per horizon (1w, 1m, 3m, 6m)
"""
import os
import time
import argparse
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import logging
import joblib
import duckdb

logging.basicConfig(level=logging.INFO)

# Use relative paths from project root
ROOT_DIR = Path(__file__).resolve().parents[3]  # Go up from src/training/baselines/
MODELS_DIR = ROOT_DIR / "models" / "baselines" / "lightgbm"

# Create directories if they don't exist (on first run)
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    # Running in read-only mode or sandbox - directories will be created on actual training
    pass

HORIZONS = {
    "target_ret_1w": "1w",
    "target_ret_1m": "1m",
    "target_ret_3m": "3m",
    "target_ret_6m": "6m",
}


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def _load_local_env() -> None:
    _load_dotenv_file(ROOT_DIR / ".env")
    _load_dotenv_file(ROOT_DIR / ".env.local")


def _iter_motherduck_tokens():
    candidates = [
        ("MOTHERDUCK_TOKEN", os.getenv("MOTHERDUCK_TOKEN")),
        ("motherduck_storage_MOTHERDUCK_TOKEN", os.getenv("motherduck_storage_MOTHERDUCK_TOKEN")),
        ("MOTHERDUCK_READ_SCALING_TOKEN", os.getenv("MOTHERDUCK_READ_SCALING_TOKEN")),
        (
            "motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN",
            os.getenv("motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN"),
        ),
    ]
    for name, value in candidates:
        if not value:
            continue
        token = value.strip().strip('"').strip("'")
        if token.count(".") != 2:
            continue
        yield name, token


def _connect_motherduck() -> duckdb.DuckDBPyConnection:
    _load_local_env()
    db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    last_error: Exception | None = None
    for _, token in _iter_motherduck_tokens():
        try:
            con = duckdb.connect(f"md:{db_name}?motherduck_token={token}")
            con.execute("SELECT 1").fetchone()
            return con
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to connect to MotherDuck: {last_error}")


def _connect_local_duckdb() -> duckdb.DuckDBPyConnection | None:
    local_path = ROOT_DIR / "data" / "duckdb" / "cbi_v15.duckdb"
    if not local_path.exists():
        return None
    try:
        con = duckdb.connect(str(local_path), read_only=True)
        con.execute("SELECT 1").fetchone()
        try:
            con.execute("SELECT 1 FROM training.daily_ml_matrix_zl LIMIT 1").fetchone()
        except Exception:
            con.close()
            return None
        return con
    except Exception:
        return None


def load_training_matrix() -> pd.DataFrame:
    """
    Prefer local synced DuckDB if available; otherwise pull from MotherDuck.
    """
    con = _connect_local_duckdb()
    src = "local"
    if con is None:
        con = _connect_motherduck()
        src = "motherduck"

    logging.info(f"Loading training.daily_ml_matrix_zl from {src}...")
    df = con.execute(
        """
        SELECT *
        FROM training.daily_ml_matrix_zl
        WHERE symbol = 'ZL'
        ORDER BY as_of_date
        """
    ).fetchdf()
    try:
        con.close()
    except Exception:
        pass
    return df


def train_lgbm_for_horizon(horizon_col: str, horizon_name: str):
    """Train LightGBM model for a specific horizon"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Training LightGBM for {horizon_name} ({horizon_col})")
    logging.info(f"{'='*60}")

    df = load_training_matrix()

    train = df[df["train_val_test_split"] == "train"].copy()
    val = df[df["train_val_test_split"] == "val"].copy()
    test = df[df["train_val_test_split"] == "test"].copy()

    # Filter out rows with missing target
    train = train[train[horizon_col].notna()].copy()
    val = val[val[horizon_col].notna()].copy()
    test = test[test[horizon_col].notna()].copy()

    logging.info(f"Train: {len(train):,} rows")
    logging.info(f"Val: {len(val):,} rows")
    logging.info(f"Test: {len(test):,} rows")

    # Drop non-feature columns
    meta_cols = [
        "as_of_date",
        "symbol",
        "train_val_test_split",
        "updated_at",
        "training_weight",
    ]
    target = horizon_col

    # Get all target columns to exclude
    all_targets = [
        "target_price_1w",
        "target_price_1m",
        "target_price_3m",
        "target_price_6m",
        "target_ret_1w",
        "target_ret_1m",
        "target_ret_3m",
        "target_ret_6m",
        "target_ret_12m",
    ]

    feature_cols = sorted(
        col for col in train.columns if col not in meta_cols + all_targets
    )

    logging.info(f"Features: {len(feature_cols)}")

    # Prepare data
    X_train = train[feature_cols].fillna(0)  # Simple null handling for now
    y_train = train[target]
    X_val = val[feature_cols].fillna(0)
    y_val = val[target]
    X_test = test[feature_cols].fillna(0)
    y_test = test[target]

    # Ensure numeric dtypes (DuckDB DECIMAL can land as object)
    for frame in (X_train, X_val, X_test):
        for c in frame.columns:
            if frame[c].dtype == "object":
                frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0)

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
        "force_row_wise": True,
        "seed": 42,
    }

    # Train model
    logging.info("Training model...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=_TRAIN_ARGS.num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(_TRAIN_ARGS.early_stopping_rounds),
            lgb.log_evaluation(_TRAIN_ARGS.log_eval),
        ],
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
    logging.info(
        f"Train - MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}"
    )
    logging.info(f"Val   - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}")
    logging.info(
        f"Test  - MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}"
    )

    # Save model
    model_path = MODELS_DIR / f"lightgbm_zl_{horizon_name}.pkl"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "horizon": horizon_name,
            "metrics": {
                "train": {"mae": mae_train, "rmse": rmse_train, "r2": r2_train},
                "val": {"mae": mae_val, "rmse": rmse_val, "r2": r2_val},
                "test": {"mae": mae_test, "rmse": rmse_test, "r2": r2_test},
            },
        },
        model_path,
    )
    logging.info(f"✅ Model saved to {model_path}")

    # Log to MotherDuck ops tables (best-effort, non-blocking)
    try:
        md = _connect_motherduck()
        run_id = f"lgbm_zl_{horizon_name}_{time.strftime('%Y%m%d_%H%M%S')}"

        val_dir = (
            float(np.mean(np.sign(y_val.values) == np.sign(y_pred_val))) if len(y_val) else None
        )
        test_dir = (
            float(np.mean(np.sign(y_test.values) == np.sign(y_pred_test))) if len(y_test) else None
        )

        md.execute(
            """
            INSERT INTO ops.training_runs (
              run_id, run_timestamp, model_tier, model_name, bucket, horizon_code,
              training_start_date, training_end_date, n_train_rows, n_val_rows,
              hyperparameters, val_mape, val_rmse, val_directional_accuracy,
              training_time_seconds, status, error_message, artifact_uri, created_at
            ) VALUES (
              ?, CURRENT_TIMESTAMP, 'core', 'lightgbm_zl', NULL, ?,
              ?, ?, ?, ?,
              ?, NULL, ?, ?,
              NULL, 'completed', NULL, ?, CURRENT_TIMESTAMP
            )
            """,
            [
                run_id,
                horizon_name,
                str(train["as_of_date"].min()) if len(train) else None,
                str(train["as_of_date"].max()) if len(train) else None,
                int(len(train)),
                int(len(val)),
                {"params": params, "best_iteration": int(model.best_iteration or 0)},
                float(rmse_val),
                float(val_dir) if val_dir is not None else None,
                str(model_path),
            ],
        )

        md.execute(
            """
            DELETE FROM reference.model_registry WHERE model_id = ?
            """,
            [f"lightgbm_zl_{horizon_name}"],
        )

        md.execute(
            """
            INSERT INTO reference.model_registry (
              model_id, model_tier, model_name, bucket, horizon_code,
              mape, directional_accuracy, coverage_90,
              ensemble_weight, is_active, artifact_uri, version, created_at, updated_at
            ) VALUES (
              ?, 'core', 'lightgbm_zl', NULL, ?,
              NULL, ?, NULL,
              NULL, TRUE, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )
            """,
            [
                f"lightgbm_zl_{horizon_name}",
                horizon_name,
                float(test_dir) if test_dir is not None else None,
                str(model_path),
            ],
        )

        md.close()
        logging.info("✅ Logged run/model to MotherDuck (ops.training_runs, reference.model_registry)")
    except Exception as e:
        logging.warning(f"⚠️  Could not log training run to MotherDuck: {e}")

    return model, feature_cols


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM baselines from training.daily_ml_matrix_zl")
    p.add_argument(
        "--horizons",
        default="1w,1m,3m,6m",
        help="Comma-separated horizon codes to train (default: 1w,1m,3m,6m)",
    )
    p.add_argument("--num-boost-round", dest="num_boost_round", type=int, default=5000)
    p.add_argument("--early-stopping-rounds", dest="early_stopping_rounds", type=int, default=200)
    p.add_argument("--log-eval", dest="log_eval", type=int, default=100)
    return p.parse_args()


_TRAIN_ARGS = _parse_args()


if __name__ == "__main__":
    logging.info("Starting LightGBM baseline training for ZL...")

    models = {}
    want = {h.strip() for h in str(_TRAIN_ARGS.horizons).split(",") if h.strip()}
    for horizon_col, horizon_name in HORIZONS.items():
        if horizon_name not in want:
            continue
        model, feat_cols = train_lgbm_for_horizon(horizon_col, horizon_name)
        models[horizon_name] = (model, feat_cols)

    logging.info("\n✅ All models trained successfully!")
    logging.info(f"Models saved to: {MODELS_DIR}")
