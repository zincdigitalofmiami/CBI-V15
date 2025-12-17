#!/usr/bin/env python3
"""
L0 Bucket Specialist Training with AutoGluon TabularPredictor

This script trains 8 specialized models (one per Big 8 bucket) using
feature-selected subsets of the training data. Each specialist:
- Uses AutoGluon TabularPredictor with extreme_quality preset
- Trains on bucket-specific features from bucket_feature_selectors.yaml
- Produces quantile forecasts (P10, P50, P90) for 4 horizons
- Saves Out-of-Fold (OOF) predictions to training.bucket_predictions

Architecture:
- L0: 8 Bucket Specialists (this script)
- L1: Meta-model (future - reads OOF predictions)
- L2: Final ensemble
- L3: Monte Carlo simulation

Expected runtime: 2-4 hours on Mac M4 with extreme_quality preset
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv

# Load Environment
load_dotenv()
DB_NAME = os.getenv("MOTHERDUCK_DB", "cbi_v15")
TOKEN = os.getenv("MOTHERDUCK_TOKEN")

# Constants
BUCKETS = [
    "crush",
    "china",
    "fx",
    "fed",
    "tariff",
    "biofuel",
    "energy",
    "volatility",
]
HORIZONS = [
    "target_price_1w",
    "target_price_1m",
    "target_price_3m",
    "target_price_6m",
]
HORIZON_CODES = {
    "target_price_1w": "1w",
    "target_price_1m": "1m",
    "target_price_3m": "3m",
    "target_price_6m": "6m",
}


def get_connection():
    """Get MotherDuck connection"""
    if not TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN not set in environment")
    return duckdb.connect(f"md:{DB_NAME}?motherduck_token={TOKEN}")


def load_data(con):
    """Load training matrix from MotherDuck"""
    print("⏳ Loading training matrix from MotherDuck...")
    df = con.execute(
        "SELECT * FROM training.daily_ml_matrix_zl ORDER BY as_of_date"
    ).df()
    print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Ensure date is datetime
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df


def get_feature_subset(
    bucket_name: str, config: Dict, available_columns: List[str]
) -> List[str]:
    """
    Selects features for a specific bucket.
    Gracefully handles missing columns by warning and skipping them.

    Args:
        bucket_name: Name of the bucket (e.g., 'crush', 'china')
        config: Loaded YAML config
        available_columns: List of actual column names in the data

    Returns:
        List of feature column names for this bucket
    """
    # Get base features (used by all buckets)
    base = config.get("core_base", [])
    # Get bucket-specific features
    specific = config.get(bucket_name, [])

    wanted = set(base + specific)
    available = set(available_columns)

    # Intersection - only use features that actually exist
    final_features = sorted(list(wanted.intersection(available)))
    missing = wanted - available

    if missing:
        print(
            f"   ⚠️  Warning: {bucket_name} missing {len(missing)} features: {list(missing)[:5]}..."
        )

    return final_features


def save_oof_predictions(con, oof_df: pd.DataFrame, bucket: str, horizon_col: str):
    """
    Saves Out-Of-Fold predictions to training.bucket_predictions table.

    Args:
        con: DuckDB connection
        oof_df: DataFrame with as_of_date and quantile predictions
        bucket: Bucket name (e.g., 'crush')
        horizon_col: Target column name (e.g., 'target_price_1w')
    """
    horizon_code = HORIZON_CODES[horizon_col]

    # Prepare DataFrame for storage
    save_df = oof_df[["as_of_date"]].copy()
    save_df["bucket"] = bucket
    save_df["horizon_code"] = horizon_code
    save_df["prediction_type"] = "oof"

    # AutoGluon outputs quantiles with column names like '0.1', '0.5', '0.9'
    # Map them to q10, q50, q90
    quantile_cols = [
        c for c in oof_df.columns if c in [0.1, 0.5, 0.9, "0.1", "0.5", "0.9"]
    ]

    if len(quantile_cols) >= 3:
        save_df["q10"] = oof_df[quantile_cols[0]].values
        save_df["q50"] = oof_df[quantile_cols[1]].values
        save_df["q90"] = oof_df[quantile_cols[2]].values
    else:
        # Fallback: if quantile columns not found, use the prediction column
        print(f"      Warning: Quantile columns not found, using single prediction")
        pred_col = [c for c in oof_df.columns if c not in ["as_of_date"]][0]
        save_df["q10"] = oof_df[pred_col].values
        save_df["q50"] = oof_df[pred_col].values
        save_df["q90"] = oof_df[pred_col].values

    save_df["created_at"] = pd.Timestamp.now()

    # Create table if not exists
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS training.bucket_predictions (
            as_of_date DATE,
            bucket VARCHAR,
            horizon_code VARCHAR,
            prediction_type VARCHAR,
            q10 DOUBLE,
            q50 DOUBLE,
            q90 DOUBLE,
            created_at TIMESTAMP,
            PRIMARY KEY (as_of_date, bucket, horizon_code, prediction_type)
        )
    """
    )

    # Insert
    print(
        f"      💾 Saving {len(save_df):,} OOF predictions for {bucket}::{horizon_code}..."
    )
    con.register("oof_staging", save_df)
    con.execute(
        """
        INSERT OR REPLACE INTO training.bucket_predictions 
        SELECT * FROM oof_staging
    """
    )
    con.unregister("oof_staging")


def main():
    parser = argparse.ArgumentParser(
        description="Train L0 Bucket Specialists with AutoGluon"
    )
    parser.add_argument(
        "--config",
        default="config/bucket_feature_selectors.yaml",
        help="Path to feature selector config",
    )
    parser.add_argument(
        "--preset",
        default="medium_quality",
        help="AutoGluon preset (good_quality, medium_quality, high_quality, extreme_quality)",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=600,
        help="Time limit per model in seconds (default: 600s = 10min)",
    )
    parser.add_argument(
        "--save-oof", action="store_true", help="Save OOF predictions to MotherDuck"
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=BUCKETS,
        help="Specific buckets to train (default: all 8)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=HORIZONS,
        help="Specific horizons to train (default: all 4)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("L0 BUCKET SPECIALIST TRAINING")
    print("=" * 80)
    print(f"Preset:      {args.preset}")
    print(f"Time limit:  {args.time_limit}s per model")
    print(f"Buckets:     {len(args.buckets)} ({', '.join(args.buckets)})")
    print(f"Horizons:    {len(args.horizons)}")
    print(f"Save OOF:    {args.save_oof}")
    print("=" * 80)
    print()

    # 1. Setup
    con = get_connection()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    full_df = load_data(con)

    # 2. Splits (Time-based)
    train_df = full_df[full_df["as_of_date"] < "2023-01-01"].copy()
    val_df = full_df[
        (full_df["as_of_date"] >= "2023-01-01") & (full_df["as_of_date"] < "2024-01-01")
    ].copy()

    print(f"📊 Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")
    print()

    # 3. Training Loop
    total_models = len(args.buckets) * len(args.horizons)
    start_time = time.time()
    models_trained = 0

    for i, bucket in enumerate(args.buckets):
        print(f"\n[{i+1}/{len(args.buckets)}] 🧊 Processing Bucket: {bucket.upper()}")

        # Get features for this bucket
        features = get_feature_subset(bucket, config, full_df.columns)
        if not features:
            print(f"   ❌ No features found for {bucket}. Skipping.")
            continue

        print(f"   ✅ Selected {len(features)} features")

        for j, horizon in enumerate(args.horizons):
            if horizon not in full_df.columns:
                print(f"   ⚠️  Target {horizon} not found in data, skipping")
                continue

            print(
                f"   [{j+1}/{len(args.horizons)}] 🎯 Horizon: {HORIZON_CODES[horizon]}"
            )

            # Prepare data (Features + Target)
            train_data = train_df[features + [horizon]].copy()
            val_data = val_df[features + [horizon]].copy()

            # Drop rows where target is NaN
            train_data = train_data.dropna(subset=[horizon])
            val_data = val_data.dropna(subset=[horizon])

            print(f"      Train: {len(train_data):,} | Val: {len(val_data):,}")

            save_path = f"models/bucket_specialists/{bucket}/{HORIZON_CODES[horizon]}"

            # Initialize Predictor
            try:
                predictor = TabularPredictor(
                    label=horizon,
                    problem_type="quantile",
                    quantile_levels=[0.1, 0.5, 0.9],
                    path=save_path,
                    eval_metric="pinball_loss",
                    verbosity=2,
                )

                # Fit
                print(f"      🚀 Training with {args.preset} preset...")
                predictor.fit(
                    train_data,
                    presets=args.preset,
                    time_limit=args.time_limit,
                    num_gpus=0,  # Mac M4 CPU only
                )

                models_trained += 1

                # OOF Generation
                if args.save_oof and len(val_data) > 0:
                    print(f"      📊 Generating OOF predictions...")
                    oof_preds = predictor.predict(val_data[features])

                    # Combine with dates
                    oof_result = val_df.loc[val_data.index, ["as_of_date"]].copy()

                    # Handle different AutoGluon output formats
                    if isinstance(oof_preds, pd.DataFrame):
                        oof_result = pd.concat(
                            [
                                oof_result.reset_index(drop=True),
                                oof_preds.reset_index(drop=True),
                            ],
                            axis=1,
                        )
                    else:
                        oof_result["prediction"] = oof_preds

                    save_oof_predictions(con, oof_result, bucket, horizon)

                print(f"      ✅ Model saved to {save_path}")

            except Exception as e:
                print(f"      ❌ Error training {bucket}::{horizon}: {e}")
                continue

    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Models trained:  {models_trained}/{total_models}")
    print(f"Total time:      {elapsed/3600:.2f} hours")
    print(f"Avg per model:   {elapsed/max(models_trained, 1)/60:.1f} minutes")
    print("=" * 80)

    con.close()


if __name__ == "__main__":
    main()

