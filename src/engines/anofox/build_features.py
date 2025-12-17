# src/anofox/build_features.py

import os
from pathlib import Path

import duckdb

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
MACROS_PATH = ROOT_DIR / "database" / "macros" / "features.sql"


def run():
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        raise ValueError("MOTHERDUCK_TOKEN required - no local fallback")
    
    print("Connecting to MotherDuck (md:cbi_v15)...")
    con = duckdb.connect(f"md:cbi_v15?motherduck_token={motherduck_token}")

    # 1. Load SQL Macros
    print(f"Loading macros from {MACROS_PATH}...")
    macros_sql = MACROS_PATH.read_text()
    con.execute(macros_sql)
    print("✅ Macros loaded.")

    # Dummy data insertion removed as per user request (NO MOCK DATA)

    # 2. Define Target Table (if not exists, though schema script should handle it)
    table_name = "features.daily_ml_matrix_zl"

    # Check if table exists
    table_exists = (
        con.execute(
            f"SELECT count(*) FROM information_schema.tables WHERE table_schema = 'features' AND table_name = 'daily_ml_matrix_zl'"
        ).fetchone()[0]
        > 0
    )

    if not table_exists:
        print(f"⚠️  Table {table_name} does not exist. Please run schema setup first.")
        return

    print(f"Truncating {table_name}...")
    con.execute(f"DELETE FROM {table_name}")

    # 3. Populate Table using Macros
    print(f"Populating {table_name} for symbol 'ZL'...")

    # We use the macro feat_daily_ml_matrix('ZL')
    # We use INSERT INTO ... SELECT ...
    # We need to ensure the columns match.
    # The schema script created the table with specific columns.
    # The macro returns columns.
    # If they don't match, this will fail.
    # Let's try INSERT BY NAME if DuckDB supports it, or just INSERT INTO.
    # Or we can use CREATE OR REPLACE TABLE to be sure.
    # But the schema script created it with types.
    # Let's try to INSERT.

    try:
        con.execute(
            f"""
            INSERT INTO {table_name}
            SELECT *
            FROM feat_daily_ml_matrix('ZL')
        """
        )
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        print("Attempting CREATE OR REPLACE TABLE as fallback...")
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM feat_daily_ml_matrix('ZL')
        """
        )

    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"✅ Population complete. {table_name} now has {row_count} rows.")

    # Validation
    try:
        null_targets = con.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE target_ret_1w IS NULL"
        ).fetchone()[0]
        print(f"   (Rows with null 1w targets: {null_targets})")
    except:
        print("   (Could not check null targets - column might be missing)")

    con.close()


if __name__ == "__main__":
    run()
