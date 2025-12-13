"""
AnoFox: Build Training Matrix

Adds targets and train/val/test splits to feature matrix.
"""

import os
from pathlib import Path

import duckdb

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def build_training(con: duckdb.DuckDBPyConnection = None) -> None:
    """
    Build training.daily_ml_matrix_zl from features + targets.
    """
    if con is None:
        motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
        if motherduck_token:
            con = duckdb.connect(
                f"md:{MOTHERDUCK_DB}?motherduck_token={motherduck_token}"
            )
        else:
            # Fallback to local or error
            # For now, let's assume local if no token, consistent with build_features
            ROOT_DIR = Path(__file__).resolve().parents[3]
            DB_PATH = ROOT_DIR / "data" / "duckdb" / "cbi_v15.duckdb"
            con = duckdb.connect(str(DB_PATH))

    print("Building training matrix...")

    # Create training matrix dynamically to include all features
    con.execute(
        """
        CREATE OR REPLACE TABLE training.daily_ml_matrix_zl AS
        SELECT 
            f.*,
            -- Splits & Weights
            CASE 
                WHEN as_of_date < '2023-01-01' THEN 'train'
                WHEN as_of_date < '2024-01-01' THEN 'val'
                ELSE 'test'
            END AS train_val_test_split,
            1.0 AS training_weight,
            
            -- Targets (forward returns)
            LEAD(close, 5) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_1w,
            LEAD(close, 21) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_1m,
            LEAD(close, 63) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_3m,
            LEAD(close, 126) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_6m,
            LEAD(close, 252) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_12m
        FROM features.daily_ml_matrix_zl f
        LEFT JOIN staging.ohlcv_daily o USING (as_of_date, symbol)
    """
    )

    print("Training build complete.")


if __name__ == "__main__":
    build_training()
