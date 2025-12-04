"""
AnoFox: Build Features

Transforms staging data into the denormalized feature matrix.
Implements Big 8 bucket logic and neural score calculations.
"""

import duckdb
import os
from pathlib import Path

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")


def build_features(con: duckdb.DuckDBPyConnection = None) -> None:
    """
    Build features.daily_ml_matrix_zl_v15 from staging tables.
    Uses macros defined in database/macros/features.sql.
    """
    if con is None:
        con = duckdb.connect(f"md:{MOTHERDUCK_DB}")

    # Load macros
    macros_path = Path(__file__).parents[2] / "database" / "macros" / "features.sql"
    if macros_path.exists():
        con.execute(macros_path.read_text())

    # Build feature matrix using macros
    print("Building feature matrix...")
    con.execute(
        """
        INSERT OR REPLACE INTO features.daily_ml_matrix_zl_v15
        SELECT * FROM ano_zl_feature_matrix_v15()
    """
    )

    print("Feature build complete.")


if __name__ == "__main__":
    build_features()
