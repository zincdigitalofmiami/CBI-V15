#!/usr/bin/env python3
"""
Database Cleanup Script

Completely wipes and recreates the database from scratch.
This eliminates any legacy tables, schemas, or macros.

⚠️  WARNING: This will DELETE ALL DATA in the database!

Usage:
    python scripts/cleanup_database.py --motherduck --confirm
    python scripts/cleanup_database.py --local --confirm
    python scripts/cleanup_database.py --both --confirm
"""

import duckdb
import os
import argparse
from pathlib import Path
from datetime import datetime

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
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def list_current_state(con: duckdb.DuckDBPyConnection, target_name: str):
    """List current schemas and tables"""
    print(f"\n{'=' * 80}")
    print(f"CURRENT STATE: {target_name}")
    print(f"{'=' * 80}")

    # List schemas
    print("\n📦 SCHEMAS TO BE DELETED:")
    schemas = con.execute(
        """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT LIKE 'pg_%' 
        AND schema_name NOT IN ('information_schema', 'main')
        ORDER BY schema_name
    """
    ).fetchall()

    if schemas:
        for schema in schemas:
            print(f"  ❌ {schema[0]}")
    else:
        print("  (none)")

    # List tables
    print("\n📊 TABLES TO BE DELETED:")
    tables = con.execute(
        """
        SELECT table_schema, table_name, 
               (SELECT COUNT(*) FROM {}.{}) as row_count
        FROM information_schema.tables 
        WHERE table_schema NOT LIKE 'pg_%' 
        AND table_schema NOT IN ('information_schema', 'main')
        ORDER BY table_schema, table_name
    """.format(
            "table_schema", "table_name"
        )
    ).fetchall()

    if tables:
        for schema, table, row_count in tables:
            print(f"  ❌ {schema}.{table} ({row_count:,} rows)")
    else:
        print("  (none)")

    return len(schemas), len(tables)


def cleanup_database(con: duckdb.DuckDBPyConnection, target_name: str):
    """Drop all schemas and tables"""
    print(f"\n{'=' * 80}")
    print(f"CLEANING UP: {target_name}")
    print(f"{'=' * 80}")

    # Get all schemas to drop
    schemas = con.execute(
        """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT LIKE 'pg_%' 
        AND schema_name NOT IN ('information_schema', 'main')
        ORDER BY schema_name
    """
    ).fetchall()

    # Drop each schema with CASCADE
    for schema in schemas:
        schema_name = schema[0]
        print(f"\n🗑️  Dropping schema: {schema_name}")
        try:
            con.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
            print(f"  ✅ Dropped {schema_name}")
        except Exception as e:
            print(f"  ❌ Error dropping {schema_name}: {e}")

    print(f"\n✅ Cleanup complete for {target_name}")


def verify_cleanup(con: duckdb.DuckDBPyConnection, target_name: str):
    """Verify database is empty"""
    print(f"\n{'=' * 80}")
    print(f"VERIFYING CLEANUP: {target_name}")
    print(f"{'=' * 80}")

    # Check schemas
    schemas = con.execute(
        """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT LIKE 'pg_%' 
        AND schema_name NOT IN ('information_schema', 'main')
    """
    ).fetchall()

    # Check tables
    tables = con.execute(
        """
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema NOT LIKE 'pg_%' 
        AND table_schema NOT IN ('information_schema', 'main')
    """
    ).fetchone()[0]

    if len(schemas) == 0 and tables == 0:
        print("✅ Database is completely clean")
        return True
    else:
        print(f"⚠️  Warning: {len(schemas)} schemas and {tables} tables still exist")
        return False


def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(
        description="Clean up CBI-V15 database (DELETES ALL DATA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  WARNING: This script will DELETE ALL DATA in the database!

This includes:
  - All schemas (raw, staging, features, training, forecasts, reference, ops, tsci)
  - All tables
  - All data
  - All macros

After running this, you must run:
  python scripts/setup_database.py --both

to recreate the database structure.
        """,
    )
    parser.add_argument("--motherduck", action="store_true", help="Clean MotherDuck")
    parser.add_argument("--local", action="store_true", help="Clean local DuckDB")
    parser.add_argument("--both", action="store_true", help="Clean both")
    parser.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="Required flag to confirm you want to delete everything",
    )

    args = parser.parse_args()

    if not args.confirm:
        print("❌ Error: --confirm flag is required to proceed")
        print("This ensures you understand this will DELETE ALL DATA")
        return

    # Default to both if no option specified
    if not (args.motherduck or args.local or args.both):
        args.both = True

    targets = []
    if args.motherduck or args.both:
        targets.append(("MotherDuck", get_motherduck_connection))
    if args.local or args.both:
        targets.append(("Local DuckDB", get_local_connection))

    print("=" * 80)
    print("⚠️  DATABASE CLEANUP - DELETE ALL DATA")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for target_name, get_connection in targets:
        try:
            con = get_connection()

            # Step 1: Show current state
            schema_count, table_count = list_current_state(con, target_name)

            if schema_count == 0 and table_count == 0:
                print(f"\n✅ {target_name} is already clean (nothing to delete)")
                con.close()
                continue

            # Step 2: Final confirmation
            print(f"\n{'=' * 80}")
            print(f"⚠️  FINAL CONFIRMATION")
            print(f"{'=' * 80}")
            print(f"Target: {target_name}")
            print(f"Schemas to delete: {schema_count}")
            print(f"Tables to delete: {table_count}")
            print(f"\nThis will DELETE ALL DATA in {target_name}!")

            response = input(f"\nType 'DELETE ALL' to confirm: ")

            if response != "DELETE ALL":
                print(f"❌ Cleanup cancelled for {target_name}")
                con.close()
                continue

            # Step 3: Cleanup
            cleanup_database(con, target_name)

            # Step 4: Verify
            verify_cleanup(con, target_name)

            con.close()

        except Exception as e:
            print(f"\n❌ Error cleaning {target_name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("✅ CLEANUP COMPLETE")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📋 Next steps:")
    print("  1. Recreate database: python scripts/setup_database.py --both")
    print("  2. Ingest raw data: python trigger/DataBento/Scripts/collect_daily.py")
    print("  3. Build features: python src/engines/anofox/build_all_features.py")


if __name__ == "__main__":
    main()
