#!/usr/bin/env python3
"""
Generate OOF predictions from trained bucket specialist models.

This script loads all 32 trained models and generates Out-of-Fold predictions
on the validation data, then saves them to training.bucket_predictions.
"""

import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("MOTHERDUCK_DB", "cbi_v15")
TOKEN = os.getenv("MOTHERDUCK_TOKEN")

BUCKETS = ["crush", "china", "fx", "fed", "tariff", "biofuel", "energy", "volatility"]
HORIZONS = ["target_price_1w", "target_price_1m", "target_price_3m", "target_price_6m"]
HORIZON_CODES = {
    "target_price_1w": "1w",
    "target_price_1m": "1m",
    "target_price_3m": "3m",
    "target_price_6m": "6m",
}


def main():
    print("=" * 80)
    print("GENERATING OOF PREDICTIONS FROM TRAINED MODELS")
    print("=" * 80)
    print()

    # Connect to MotherDuck
    con = duckdb.connect(f"md:{DB_NAME}?motherduck_token={TOKEN}")

    # Load validation data
    print("📥 Loading validation data...")
    val_df = con.execute(
        """
        SELECT * FROM training.daily_ml_matrix_zl 
        WHERE as_of_date >= '2023-01-01' AND as_of_date < '2024-01-01'
        ORDER BY as_of_date
    """
    ).df()
    print(f"✅ Loaded {len(val_df):,} validation rows")
    print()

    # Load config
    with open("config/bucket_feature_selectors.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create predictions table
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

    # Clear existing OOF predictions
    con.execute("DELETE FROM training.bucket_predictions WHERE prediction_type = 'oof'")

    total_predictions = 0

    # Generate predictions for each model
    for i, bucket in enumerate(BUCKETS):
        print(f"[{i+1}/{len(BUCKETS)}] 🧊 {bucket.upper()}")

        # Get features for this bucket
        base = config.get("core_base", [])
        specific = config.get(bucket, [])
        features = list(set(base + specific).intersection(set(val_df.columns)))

        for horizon in HORIZONS:
            model_path = f"models/bucket_specialists/{bucket}/{HORIZON_CODES[horizon]}"

            if not Path(model_path).exists():
                print(f"   ⚠️  {HORIZON_CODES[horizon]}: Model not found, skipping")
                continue

            try:
                # Load model
                predictor = TabularPredictor.load(model_path, verbosity=0)

                # Prepare validation data
                val_data = val_df[features].copy()

                # Generate predictions
                preds = predictor.predict(val_data)

                # Create result DataFrame
                result_df = pd.DataFrame()
                result_df["as_of_date"] = val_df["as_of_date"].values
                result_df["bucket"] = bucket
                result_df["horizon_code"] = HORIZON_CODES[horizon]
                result_df["prediction_type"] = "oof"

                # Handle quantile predictions
                if isinstance(preds, pd.DataFrame):
                    # Quantile predictions returned as DataFrame
                    result_df["q10"] = preds.iloc[:, 0].values
                    result_df["q50"] = preds.iloc[:, 1].values
                    result_df["q90"] = preds.iloc[:, 2].values
                else:
                    # Single prediction (shouldn't happen with quantile mode)
                    result_df["q10"] = preds
                    result_df["q50"] = preds
                    result_df["q90"] = preds

                # Add missing columns from DDL schema
                result_df["p_up"] = None
                result_df["p_down"] = None
                result_df["expected_return"] = None
                result_df["confidence"] = None
                result_df["model_version"] = "v1.0"
                result_df["created_at"] = pd.Timestamp.now()

                # Save to database
                con.register("oof_temp", result_df)
                con.execute(
                    "INSERT INTO training.bucket_predictions SELECT * FROM oof_temp"
                )
                con.unregister("oof_temp")

                total_predictions += len(result_df)
                print(
                    f"   ✅ {HORIZON_CODES[horizon]}: {len(result_df):,} predictions saved"
                )

            except Exception as e:
                print(f"   ❌ {HORIZON_CODES[horizon]}: Error - {e}")

    print()
    print("=" * 80)
    print("✅ OOF PREDICTION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total predictions saved: {total_predictions:,}")
    print()

    # Final verification
    final_count = con.execute(
        "SELECT COUNT(*) FROM training.bucket_predictions"
    ).fetchone()[0]
    buckets_saved = con.execute(
        "SELECT COUNT(DISTINCT bucket) FROM training.bucket_predictions"
    ).fetchone()[0]

    print(
        f"✅ Verification: {final_count:,} predictions in database ({buckets_saved} buckets)"
    )

    con.close()


if __name__ == "__main__":
    main()

