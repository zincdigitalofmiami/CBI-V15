"""
AnoFox: Build Training Matrix

Adds targets and train/val/test splits to feature matrix.
"""

import duckdb
import os

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")


def build_training(con: duckdb.DuckDBPyConnection = None) -> None:
    """
    Build training.daily_ml_matrix_zl_v15 from features + targets.
    """
    if con is None:
        con = duckdb.connect(f"md:{MOTHERDUCK_DB}")

    print("Building training matrix...")

    # Copy features and add targets
    con.execute(
        """
        INSERT OR REPLACE INTO training.daily_ml_matrix_zl_v15
        SELECT 
            f.*,
            -- Targets (forward returns)
            LEAD(close, 5) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_1w,
            LEAD(close, 21) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_1m,
            LEAD(close, 63) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_3m,
            LEAD(close, 126) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_6m,
            LEAD(close, 252) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_12m
        FROM features.daily_ml_matrix_zl_v15 f
        LEFT JOIN staging.ohlcv_daily o USING (as_of_date, symbol)
    """
    )

    print("Training build complete.")


if __name__ == "__main__":
    build_training()
