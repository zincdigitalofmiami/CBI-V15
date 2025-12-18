#!/usr/bin/env python3
"""
Database Setup Script

Deploys all schemas, tables, macros, and views to MotherDuck/DuckDB.
Reads DDL files from database/ddl/ in numeric order.

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
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR / ".env.local", override=True)
except ImportError:
    pass
DATABASE_DIR = ROOT_DIR / "database"
DDL_DIR = DATABASE_DIR / "ddl"
MACROS_DIR = DATABASE_DIR / "macros"
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def _iter_motherduck_tokens():
    candidates = [
        ("MOTHERDUCK_TOKEN", os.getenv("MOTHERDUCK_TOKEN")),
        (
            "motherduck_storage_MOTHERDUCK_TOKEN",
            os.getenv("motherduck_storage_MOTHERDUCK_TOKEN"),
        ),
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
        if token and token.count(".") == 2:
            yield token


def get_motherduck_connection():
    """Get MotherDuck connection"""
    last_error: Exception | None = None
    for token in _iter_motherduck_tokens():
        try:
            con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={token}")
            con.execute("SELECT 1").fetchone()
            return con
        except Exception as e:
            last_error = e
            continue
    raise ValueError(
        "No working MotherDuck token found (checked MOTHERDUCK_TOKEN / motherduck_storage_MOTHERDUCK_TOKEN)"
        + (f"; last error: {last_error}" if last_error else "")
    )


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
            # Remove SQL comments before splitting
            lines = []
            for line in sql.split("\n"):
                stripped = line.strip()
                if not stripped.startswith("--"):
                    lines.append(line)
            clean_sql = "\n".join(lines)
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in clean_sql.split(";") if s.strip()]
            for stmt in statements:
                if stmt:
                    con.execute(stmt)
        print(f"    ✅ {file_path.name}")
        return True
    except Exception as e:
        print(f"    ❌ {file_path.name}: {e}")
        return False


def setup_schemas(con: duckdb.DuckDBPyConnection):
    """Create all schemas from ddl/00_schemas.sql"""
    print("\n📦 Creating schemas...")
    schema_file = DDL_DIR / "00_schemas.sql"
    if not schema_file.exists():
        print(f"    ❌ Schema file not found: {schema_file}")
        return False
    return execute_sql_file(con, schema_file, "Creating 9 schemas")


def setup_ddl_folder(con: duckdb.DuckDBPyConnection, folder_name: str, description: str):
    """Execute all SQL files in a DDL subfolder in numeric order"""
    folder_path = DDL_DIR / folder_name
    if not folder_path.exists():
        print(f"  ⚠️  Folder not found: {folder_path}")
        return True  # Not a failure, just skip
    
    print(f"\n📥 {description}...")
    
    # Get all .sql files sorted by name (numeric prefix)
    sql_files = sorted(folder_path.glob("*.sql"))
    if not sql_files:
        print(f"  ⚠️  No SQL files in {folder_name}")
        return True
    
    success = True
    for sql_file in sql_files:
        result = execute_sql_file(con, sql_file, f"Running {sql_file.name}")
        success = success and result
    
    return success


def load_macros(con: duckdb.DuckDBPyConnection):
    """Load all SQL macros from database/macros/"""
    print("\n🔧 Loading SQL macros...")
    
    if not MACROS_DIR.exists():
        print(f"  ⚠️  Macros directory not found: {MACROS_DIR}")
        return True
    
    # Load macros in dependency order
    macro_files = [
        "utils.sql",
        "asof_joins.sql",
        "anofox_guards.sql",
        "features.sql",
        "technical_indicators_all_symbols.sql",
        "cross_asset_features.sql",
        "big8_cot_enhancements.sql",
        "big8_bucket_features.sql",
        "master_feature_matrix.sql",
    ]
    
    success = True
    for file in macro_files:
        file_path = MACROS_DIR / file
        if file_path.exists():
            result = execute_sql_file(con, file_path, f"Loading {file}")
            success = success and result
        else:
            print(f"  ⚠️  {file} not found, skipping")
    
    return success


def verify_setup(con: duckdb.DuckDBPyConnection):
    """Verify database setup"""
    print("\n✅ Verifying setup...")

    # Check schemas
    schemas = con.execute(
        """SELECT schema_name FROM information_schema.schemata 
           WHERE schema_name IN ('raw', 'staging', 'features', 'features_dev', 
                                 'training', 'forecasts', 'reference', 'ops', 'explanations')
           ORDER BY schema_name"""
    ).fetchall()
    print(f"  Schemas: {len(schemas)}/9 found")
    for schema in schemas:
        print(f"    - {schema[0]}")

    # Check tables
    tables = con.execute(
        """SELECT table_schema, table_name
           FROM information_schema.tables
           WHERE table_schema IN ('raw', 'staging', 'features', 'training', 
                                  'forecasts', 'reference', 'ops', 'explanations')
           ORDER BY table_schema, table_name"""
    ).fetchall()
    print(f"\n  Tables: {len(tables)} found")
    
    # Group by schema
    by_schema = {}
    for schema, table in tables:
        by_schema.setdefault(schema, []).append(table)
    for schema, tbls in sorted(by_schema.items()):
        print(f"    {schema}: {len(tbls)} tables")

    return True


def cleanup_existing_schemas(con: duckdb.DuckDBPyConnection, target_name: str):
    """Drop all existing schemas before setup"""
    print(f"\n🗑️  Cleaning up existing schemas in {target_name}...")

    schemas_to_drop = ['raw', 'staging', 'features', 'features_dev', 
                       'training', 'forecasts', 'reference', 'ops', 'explanations']
    
    for schema_name in schemas_to_drop:
        try:
            con.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
            print(f"    ✅ Dropped {schema_name}")
        except Exception as e:
            print(f"    ⚠️  {schema_name}: {e}")

    print(f"  ✅ Cleanup complete")
    return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup CBI-V15 database from ddl/ folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal setup (creates if not exists)
  python scripts/setup_database.py --both

  # Force fresh setup (deletes everything first)
  python scripts/setup_database.py --both --force
        """,
    )
    parser.add_argument("--motherduck", action="store_true", help="Deploy to MotherDuck")
    parser.add_argument("--local", action="store_true", help="Deploy to local DuckDB")
    parser.add_argument("--both", action="store_true", help="Deploy to both")
    parser.add_argument("--force", action="store_true", help="Force fresh setup (drops schemas first)")
    parser.add_argument("--skip-macros", action="store_true", help="Skip loading macros")

    args = parser.parse_args()

    # Default to local if no option specified
    if not (args.motherduck or args.local or args.both):
        args.local = True

    targets = []
    if args.motherduck or args.both:
        targets.append(("MotherDuck", get_motherduck_connection))
    if args.local or args.both:
        targets.append(("Local DuckDB", get_local_connection))

    print("=" * 80)
    print("CBI-V15 DATABASE SETUP")
    print(f"DDL Source: {DDL_DIR}")
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
                cleanup_existing_schemas(con, target_name)

            # Step 1: Create schemas (00_schemas.sql)
            if not setup_schemas(con):
                print(f"❌ Failed to create schemas in {target_name}")
                continue

            # Step 2-8: Run DDL subfolders in numeric order
            ddl_folders = [
                ("01_reference", "Creating reference tables"),
                ("02_raw", "Creating raw tables"),
                ("03_staging", "Creating staging tables"),
                ("04_features", "Creating feature tables"),
                ("05_training", "Creating training tables"),
                ("06_forecasts", "Creating forecast tables"),
                ("07_ops", "Creating ops tables"),
                ("08_explanations", "Creating explanations tables"),
            ]
            
            for folder, desc in ddl_folders:
                setup_ddl_folder(con, folder, desc)

            # Step 9: Load macros (unless skipped)
            if not args.skip_macros:
                load_macros(con)

            # Step 10: Verify setup
            verify_setup(con)

            print(f"\n✅ {target_name} setup complete!")
            con.close()

        except Exception as e:
            print(f"\n❌ Error setting up {target_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✅ DATABASE SETUP COMPLETE")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("  1. Seed reference data: python database/seeds/seed_symbols.py")
    print("  2. Run features: python src/engines/anofox/build_all_features.py")
    print("  3. Train models: python src/training/autogluon/mitra_trainer.py")


if __name__ == "__main__":
    main()
