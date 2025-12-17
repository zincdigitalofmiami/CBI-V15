"""
READ-ONLY: Extract cached OOF predictions from Big 8 bucket specialists.

STRICTLY READ-ONLY - NO MODEL LOADING:
- Only reads existing cached OOF prediction CSV files
- Never loads or runs any AutoGluon models
- Skips buckets/horizons without cached predictions
- Pure data extraction from disk files

Usage:
    python scripts/training/save_the_brain.py
"""

import pandas as pd
import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION - Updated for CBI-V15 architecture
DB_NAME = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def main():
    print(f"🔌 Connecting to MotherDuck: {DB_NAME}...")
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise ValueError("❌ MOTHERDUCK_TOKEN is missing!")

    con = duckdb.connect(f"md:{DB_NAME}?motherduck_token={token}")

    # 1. VERIFY SCHEMA MATCHES DDL (DECIMAL types)
    try:
        # Check if table exists and get schema
        print("🔍 Verifying Schema...")
        schema_check = con.execute("DESCRIBE training.bucket_predictions").fetchall()
        col_types = {col[0]: col[1] for col in schema_check}

        # Check for DECIMAL types as per DDL
        if "q10" in col_types and "DECIMAL" in col_types["q10"]:
            print("✅ Verified: Schema uses DECIMAL types (matches DDL).")
        else:
            print("⚠️ WARNING: Schema mismatch detected. DDL expects DECIMAL types.")
            # Don't auto-create - let user run setup_database.py
            raise ValueError(
                "Schema doesn't match DDL. Run 'python scripts/setup_database.py --both' first."
            )

    except Exception as e:
        print(f"⚠️ Table verification failed: {e}")
        print(
            "💡 Run 'python scripts/setup_database.py --both' to create proper schema"
        )
        raise

    # 2. LOAD BUCKET SPECIALIST MODELS
    # CBI-V15 has 8 bucket specialists + 1 main ZL predictor
    # This script should extract OOF predictions from trained bucket specialists
    print("🧠 Loading Bucket Specialist Models...")

    # Big 8 buckets as defined in architecture
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
    HORIZON_CODES = ["1w", "1m", "3m", "6m"]

    all_predictions = []

    for bucket in BUCKETS:
        print(f"📂 Processing bucket: {bucket}")
        bucket_predictions = []

        for horizon in HORIZON_CODES:
            model_path = f"models/bucket_specialists/{bucket}/{horizon}"

            # STRICTLY READ-ONLY: Only read existing cached files
            oof_file = f"{model_path}/oof_predictions.csv"
            if not os.path.exists(oof_file):
                print(f"  ⏭️  Skipping {bucket}/{horizon} - no cached predictions found")
                continue

            print(f"  📖 Reading cached OOF predictions from: {oof_file}")
            df_oof = pd.read_csv(oof_file)

            # Transform to bucket_predictions schema
            for _, row in df_oof.iterrows():
                prediction = {
                    "as_of_date": row["as_of_date"],  # Assuming this column exists
                    "bucket": bucket,
                    "horizon_code": horizon,
                    "prediction_type": "oof",
                    "q10": float(row["q10"]),
                    "q50": float(row["q50"]),
                    "q90": float(row["q90"]),
                    "model_version": "v15.0.0-beta",
                    "created_at": pd.Timestamp.now(),
                }

                    # Calculate derived metrics (corrected formulas)
                    q10, q50, q90 = (
                        prediction["q10"],
                        prediction["q50"],
                        prediction["q90"],
                    )

                    # P_UP: Probability price goes up (Q90 > 0)
                    prediction["p_up"] = 1.0 if q90 > 0 else 0.0
                    prediction["p_down"] = 1.0 - prediction["p_up"]

                    # Expected Return: Simple average of quantiles
                    prediction["expected_return"] = (q10 + q50 + q90) / 3.0

                    # Confidence: Inverse of quantile spread (higher spread = lower confidence)
                    spread = q90 - q10
                    prediction["confidence"] = (
                        1.0 / (1.0 + spread) if spread > 0 else 1.0
                    )

                    bucket_predictions.append(prediction)

            except Exception as e:
                print(f"  ❌ Error loading {bucket}/{horizon}: {e}")
                continue

        all_predictions.extend(bucket_predictions)
        print(f"  ✅ Collected {len(bucket_predictions)} predictions for {bucket}")

    if not all_predictions:
        raise ValueError(
            "❌ No bucket specialist models found! Train bucket specialists first."
        )

    df = pd.DataFrame(all_predictions)
    print(f"🧮 Generated {len(df)} total predictions across all buckets")

    # 5. SAVE TO MOTHERDUCK (with DECIMAL casting)
    final_cols = [
        "as_of_date",
        "bucket",
        "horizon_code",
        "prediction_type",
        "p_up",
        "p_down",
        "expected_return",
        "confidence",
        "q10",
        "q50",
        "q90",
        "model_version",
        "created_at",
    ]
    df_final = df[final_cols].copy()

    # Ensure proper DECIMAL precision for schema compliance
    decimal_cols = ["p_up", "p_down", "confidence"]  # DECIMAL(5,4) - 4 decimal places
    quantile_cols = [
        "q10",
        "q50",
        "q90",
        "expected_return",
    ]  # DECIMAL(10,4) - 4 decimal places

    for col in decimal_cols + quantile_cols:
        df_final[col] = df_final[col].round(4)  # Round to 4 decimal places

    print(f"💾 Saving {len(df_final)} predictions to MotherDuck...")
    con.register("df_view", df_final)
    con.execute(
        "INSERT OR REPLACE INTO training.bucket_predictions SELECT * FROM df_view"
    )
    print("🎉 Success! Bucket specialist brains saved to MotherDuck.")


if __name__ == "__main__":
    main()
