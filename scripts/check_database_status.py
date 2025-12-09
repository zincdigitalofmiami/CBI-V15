#!/usr/bin/env python3
"""
Check Database Status

Checks what's currently deployed in MotherDuck/DuckDB.

Usage:
    python scripts/check_database_status.py --motherduck
    python scripts/check_database_status.py --local
    python scripts/check_database_status.py --both
"""

import argparse
import os
from pathlib import Path

import duckdb

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")
ROOT_DIR = Path(__file__).resolve().parents[1]


def get_motherduck_connection():
    """Get MotherDuck connection"""
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        raise ValueError("MOTHERDUCK_TOKEN environment variable not set")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={motherduck_token}")


def get_local_connection():
    """Get local DuckDB connection"""
    db_path = ROOT_DIR / "data" / "duckdb" / "cbi_v15.duckdb"
    if not db_path.exists():
        print(f"❌ Local database not found at {db_path}")
        return None
    return duckdb.connect(str(db_path))


def check_status(con: duckdb.DuckDBPyConnection, target_name: str):
    """Check database status"""
    print(f"\n{'=' * 80}")
    print(f"DATABASE STATUS: {target_name}")
    print(f"{'=' * 80}")

    # Check schemas
    print("\n📦 SCHEMAS:")
    schemas = con.execute(
        """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT LIKE 'pg_%' 
        AND schema_name != 'information_schema'
        ORDER BY schema_name
    """
    ).fetchall()

    if schemas:
        for schema in schemas:
            print(f"  ✅ {schema[0]}")
    else:
        print("  ❌ No schemas found")

    # Check tables by schema
    print("\n📊 TABLES:")
    for schema in ["raw", "staging", "features", "training", "forecasts", "reference"]:
        tables = con.execute(
            f"""
            SELECT table_name, 
                   (SELECT COUNT(*) FROM {schema}.{table_name}) as row_count
            FROM information_schema.tables 
            WHERE table_schema = '{schema}'
            ORDER BY table_name
        """
        ).fetchall()

        if tables:
            print(f"\n  {schema.upper()}:")
            for table, row_count in tables:
                print(f"    ✅ {table} ({row_count:,} rows)")
        else:
            print(f"\n  {schema.upper()}:")
            print(f"    ⚠️  No tables found")

    # Check macros (try to execute a test macro)
    print("\n🔧 MACROS:")
    macros_to_test = [
        ("calc_rsi", "SELECT * FROM calc_rsi('ZL', 14) LIMIT 1"),
        ("calc_macd", "SELECT * FROM calc_macd('ZL', 12, 26, 9) LIMIT 1"),
        ("calc_bollinger", "SELECT * FROM calc_bollinger('ZL', 20, 2) LIMIT 1"),
        ("calc_all_bucket_scores", "SELECT * FROM calc_all_bucket_scores() LIMIT 1"),
    ]

    for macro_name, test_query in macros_to_test:
        try:
            con.execute(test_query)
            print(f"  ✅ {macro_name}")
        except Exception as e:
            print(f"  ❌ {macro_name} - {str(e)[:50]}")

    # Check feature matrix
    print("\n🎯 MASTER FEATURE MATRIX:")
    try:
        result = con.execute(
            """
            SELECT 
                COUNT(*) as row_count,
                MIN(as_of_date) as earliest_date,
                MAX(as_of_date) as latest_date
            FROM features.daily_ml_matrix_zl
        """
        ).fetchone()

        if result and result[0] > 0:
            print(f"  ✅ features.daily_ml_matrix_zl")
            print(f"     Rows: {result[0]:,}")
            print(f"     Date range: {result[1]} to {result[2]}")

            # Count columns
            cols = con.execute(
                """
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_schema = 'features' 
                AND table_name = 'daily_ml_matrix_zl'
            """
            ).fetchone()[0]
            print(f"     Columns: {cols}")
        else:
            print(f"  ⚠️  features.daily_ml_matrix_zl exists but is empty")
    except Exception as e:
        print(f"  ❌ features.daily_ml_matrix_zl not found")

    print(f"\n{'=' * 80}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Check CBI-V15 database status")
    parser.add_argument("--motherduck", action="store_true", help="Check MotherDuck")
    parser.add_argument("--local", action="store_true", help="Check local DuckDB")
    parser.add_argument("--both", action="store_true", help="Check both")

    args = parser.parse_args()

    # Default to both if no option specified
    if not (args.motherduck or args.local or args.both):
        args.both = True

    targets = []
    if args.motherduck or args.both:
        targets.append(("MotherDuck", get_motherduck_connection))
    if args.local or args.both:
        targets.append(("Local DuckDB", get_local_connection))

    for target_name, get_connection in targets:
        try:
            con = get_connection()
            if con:
                check_status(con, target_name)
                con.close()
        except Exception as e:
            print(f"\n❌ Error connecting to {target_name}: {e}\n")


if __name__ == "__main__":
    main()
