#!/usr/bin/env python3
"""
Sync DDLs to MotherDuck
Applies all SQL schema definitions to the cloud database.
"""
import os
import glob
import duckdb
from pathlib import Path

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def get_connection():
    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN not found")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")


def main():
    print(f"🔌 Connecting to MotherDuck: {MOTHERDUCK_DB}")
    con = get_connection()

    # DDL Order matters
    ddl_dirs = [
        "00_init",
        "02_raw",
        "03_staging",
        "04_features",
        "05_training",
        "06_forecasts",
        "07_reference",
        "08_ops",
    ]

    root_ddl = Path(__file__).parents[2] / "database" / "ddl"

    for ddl_dir in ddl_dirs:
        dir_path = root_ddl / ddl_dir
        if not dir_path.exists():
            print(f"⚠️ Directory not found: {ddl_dir}")
            continue

        print(f"\n📂 Processing {ddl_dir}...")
        # Get all .sql files
        sql_files = sorted(dir_path.glob("*.sql"))

        for sql_file in sql_files:
            print(f"  📜 Executing {sql_file.name}...")
            try:
                with open(sql_file, "r") as f:
                    sql = f.read()
                    con.execute(sql)
            except Exception as e:
                print(f"    ❌ Error: {e}")
                # Don't stop, continue to next

    print("\n✅ DDL Sync Complete")


if __name__ == "__main__":
    main()




