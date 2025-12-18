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

from __future__ import annotations

import pandas as pd
import duckdb
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


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
    for _, value in candidates:
        if not value:
            continue
        token = value.strip().strip('"').strip("'")
        if token.count(".") != 2:
            continue
        yield token

# CONFIGURATION - Updated for CBI-V15 architecture
DB_NAME = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def main():
    print(f"🔌 Connecting to MotherDuck: {DB_NAME}...")
    _load_local_env()
    con = None
    last_error: Exception | None = None
    for token in _iter_motherduck_tokens():
        try:
            con = duckdb.connect(f"md:{DB_NAME}?motherduck_token={token}")
            con.execute("SELECT 1").fetchone()
            break
        except Exception as e:
            last_error = e
            con = None
    if con is None:
        raise ValueError(f"❌ MotherDuck token is missing or invalid: {last_error}")

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
            try:
                df_oof = pd.read_csv(oof_file)
                required = {"as_of_date", "q10", "q50", "q90"}
                if not required.issubset(df_oof.columns):
                    missing = sorted(required - set(df_oof.columns))
                    print(f"  ⚠️  Skipping {bucket}/{horizon} - missing columns: {missing}")
                    continue

                for _, row in df_oof.iterrows():
                    q10 = float(row["q10"])
                    q50 = float(row["q50"])
                    q90 = float(row["q90"])

                    spread = q90 - q10
                    confidence = 1.0 / (1.0 + spread) if spread > 0 else 1.0

                    prediction = {
                        "as_of_date": row["as_of_date"],
                        "bucket": bucket,
                        "horizon_code": horizon,
                        "prediction_type": "oof",
                        "p_up": 1.0 if q90 > 0 else 0.0,
                        "p_down": 0.0 if q90 > 0 else 1.0,
                        "expected_return": (q10 + q50 + q90) / 3.0,
                        "confidence": confidence,
                        "q10": q10,
                        "q50": q50,
                        "q90": q90,
                        "model_version": "v15.0.0-beta",
                        "created_at": pd.Timestamp.now(),
                    }
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
    # No PK constraints in MotherDuck tables; do an explicit delete+insert upsert.
    con.execute(
        """
        DELETE FROM training.bucket_predictions
        WHERE EXISTS (
          SELECT 1
          FROM df_view v
          WHERE v.as_of_date = training.bucket_predictions.as_of_date
            AND v.bucket = training.bucket_predictions.bucket
            AND v.horizon_code = training.bucket_predictions.horizon_code
            AND v.prediction_type = training.bucket_predictions.prediction_type
        )
        """
    )
    con.execute("INSERT INTO training.bucket_predictions SELECT * FROM df_view")
    print("🎉 Success! Bucket specialist brains saved to MotherDuck.")


if __name__ == "__main__":
    main()
