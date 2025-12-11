#!/usr/bin/env python3
"""
Database Setup Script

Deploys all schemas, tables, macros, and views to MotherDuck/DuckDB.

Usage:
    python scripts/setup_database.py --motherduck  # Deploy to MotherDuck
    python scripts/setup_database.py --local       # Deploy to local DuckDB
    python scripts/setup_database.py --both        # Deploy to both
"""

import duckdb
import os
import argparse
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT_DIR / "database"
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")


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


def execute_sql_file(con: duckdb.DuckDBPyConnection, file_path: Path, description: str):
    """Execute a SQL file"""
    print(f"  📄 {description}...")
    try:
        with open(file_path, "r") as f:
            sql = f.read()
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                if stmt:
                    con.execute(stmt)
        print(f"    ✅ {file_path.name}")
        return True
    except Exception as e:
        print(f"    ❌ {file_path.name}: {e}")
        return False


def setup_schemas(con: duckdb.DuckDBPyConnection):
    """Create all schemas"""
    print("\n📦 Creating schemas...")
    schema_file = DATABASE_DIR / "definitions" / "00_init" / "00_schemas.sql"
    return execute_sql_file(con, schema_file, "Creating 8 schemas")


def setup_raw_tables(con: duckdb.DuckDBPyConnection):
    """Create raw data tables"""
    print("\n📥 Creating raw tables...")
    raw_dir = DATABASE_DIR / "definitions" / "01_raw"

    files = [
        "databento_daily.sql",
        "fred_macro.sql",
        "eia_biofuels.sql",
        "epa_rin_prices.sql",
        "scrapecreators_buckets.sql",
        "cftc_cot.sql",  # CFTC COT tables
        "noaa_weather.sql",  # Weather data
        "usda_data.sql",  # USDA WASDE, export sales, crop progress
    ]

    success = True
    for file in files:
        file_path = raw_dir / file
        if file_path.exists():
            success &= execute_sql_file(con, file_path, f"Creating {file}")
        else:
            print(f"  ⚠️  {file} not found, skipping")

    return success


def setup_staging_tables(con: duckdb.DuckDBPyConnection):
    """Create staging tables"""
    print("\n🔄 Creating staging tables...")
    staging_dir = DATABASE_DIR / "definitions" / "02_staging"

    files = [
        "market_daily.sql",
        "crush_daily.sql",
        "china_daily.sql",
        "news_bucketed.sql",
    ]

    success = True
    for file in files:
        file_path = staging_dir / file
        if file_path.exists():
            success &= execute_sql_file(con, file_path, f"Creating {file}")
        else:
            print(f"  ⚠️  {file} not found, skipping")

    return success


def setup_feature_tables(con: duckdb.DuckDBPyConnection):
    """Create feature tables"""
    print("\n📊 Creating feature tables...")
    features_dir = DATABASE_DIR / "definitions" / "03_features"

    files = ["technical_indicators_all_symbols.sql", "daily_ml_matrix.sql"]

    success = True
    for file in files:
        file_path = features_dir / file
        if file_path.exists():
            success &= execute_sql_file(con, file_path, f"Creating {file}")
        else:
            print(f"  ⚠️  {file} not found, skipping")

    return success


def setup_training_tables(con: duckdb.DuckDBPyConnection):
    """Create training tables"""
    print("\n🎯 Creating training tables...")
    training_dir = DATABASE_DIR / "definitions" / "04_training"

    file_path = training_dir / "daily_ml_matrix.sql"
    if file_path.exists():
        return execute_sql_file(con, file_path, "Creating training tables")
    else:
        print(f"  ⚠️  daily_ml_matrix.sql not found, skipping")
        return False


def load_macros(con: duckdb.DuckDBPyConnection):
    """Load all SQL macros"""
    print("\n🔧 Loading SQL macros...")
    macros_dir = DATABASE_DIR / "macros"

    files = [
        "features.sql",
        "technical_indicators_all_symbols.sql",
        "cross_asset_features.sql",
        "big8_cot_enhancements.sql",  # COT helper macros
        "big8_bucket_features.sql",
        "master_feature_matrix.sql",
    ]

    success = True
    for file in files:
        file_path = macros_dir / file
        if file_path.exists():
            success &= execute_sql_file(con, file_path, f"Loading {file}")
        else:
            print(f"  ⚠️  {file} not found, skipping")

    return success


def verify_setup(con: duckdb.DuckDBPyConnection):
    """Verify database setup"""
    print("\n✅ Verifying setup...")

    # Check schemas
    schemas = con.execute(
        "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT LIKE 'pg_%' AND schema_name != 'information_schema'"
    ).fetchall()
    print(f"  Schemas: {len(schemas)} found")
    for schema in schemas:
        print(f"    - {schema[0]}")

    # Check tables
    tables = con.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('raw', 'staging', 'features', 'training', 'forecasts', 'reference')
        ORDER BY table_schema, table_name
    """
    ).fetchall()
    print(f"\n  Tables: {len(tables)} found")
    for schema, table in tables:
        print(f"    - {schema}.{table}")

    return True


def cleanup_existing_schemas(con: duckdb.DuckDBPyConnection, target_name: str):
    """Drop all existing schemas before setup"""
    print(f"\n🗑️  Cleaning up existing schemas in {target_name}...")

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

    if not schemas:
        print("  ✅ No existing schemas to clean up")
        return True

    # Drop each schema with CASCADE
    for schema in schemas:
        schema_name = schema[0]
        print(f"  Dropping schema: {schema_name}")
        try:
            con.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
            print(f"    ✅ Dropped {schema_name}")
        except Exception as e:
            print(f"    ❌ Error dropping {schema_name}: {e}")
            return False

    print(f"  ✅ Cleanup complete")
    return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup CBI-V15 database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal setup (creates if not exists)
  python scripts/setup_database.py --both

  # Force fresh setup (deletes everything first)
  python scripts/setup_database.py --both --force
        """,
    )
    parser.add_argument(
        "--motherduck", action="store_true", help="Deploy to MotherDuck"
    )
    parser.add_argument("--local", action="store_true", help="Deploy to local DuckDB")
    parser.add_argument(
        "--both", action="store_true", help="Deploy to both MotherDuck and local"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force fresh setup (drops all existing schemas first)",
    )

    args = parser.parse_args()

    # Default to both if no option specified
    if not (args.motherduck or args.local or args.both):
        args.both = True

    targets = []
    if args.motherduck or args.both:
        targets.append(("MotherDuck", get_motherduck_connection))
    if args.local or args.both:
        targets.append(("Local DuckDB", get_local_connection))

    print("=" * 80)
    print("CBI-V15 DATABASE SETUP")
    if args.force:
        print("⚠️  FORCE MODE: Will delete all existing schemas first")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for target_name, get_connection in targets:
        print(f"\n{'=' * 80}")
        print(f"DEPLOYING TO: {target_name}")
        print(f"{'=' * 80}")

        try:
            con = get_connection()

            # Step 0: Cleanup if --force flag is set
            if args.force:
                if not cleanup_existing_schemas(con, target_name):
                    print(f"❌ Failed to cleanup {target_name}")
                    continue

            # Step 1: Create schemas
            if not setup_schemas(con):
                print(f"❌ Failed to create schemas in {target_name}")
                continue

            # Step 2: Create raw tables
            if not setup_raw_tables(con):
                print(f"⚠️  Some raw tables failed in {target_name}")

            # Step 3: Create staging tables
            if not setup_staging_tables(con):
                print(f"⚠️  Some staging tables failed in {target_name}")

            # Step 4: Create feature tables
            if not setup_feature_tables(con):
                print(f"⚠️  Some feature tables failed in {target_name}")

            # Step 5: Create training tables
            if not setup_training_tables(con):
                print(f"⚠️  Training tables failed in {target_name}")

            # Step 6: Load macros
            if not load_macros(con):
                print(f"⚠️  Some macros failed to load in {target_name}")

            # Step 7: Verify setup
            verify_setup(con)

            print(f"\n✅ {target_name} setup complete!")

            con.close()

        except Exception as e:
            print(f"\n❌ Error setting up {target_name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("✅ DATABASE SETUP COMPLETE")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("  1. Run: python src/engines/anofox/build_all_features.py")
    print(
        "  2. Train models: python src/training/probabilistic/train_catboost_all_buckets.py"
    )
    print("  3. Generate forecasts: python src/ensemble/monte_carlo_ensemble.py")


if __name__ == "__main__":
    main()
