#!/usr/bin/env python3
"""
Sync MotherDuck → Local DuckDB
Mirrors all data from MotherDuck cloud to local DuckDB for fast training I/O.

This uses MotherDuck's hybrid query execution with ATTACH to copy tables
from cloud to local disk for 100-1000x faster reads during training.

Usage:
    python scripts/sync_motherduck_to_local.py [--schemas SCHEMA1,SCHEMA2] [--dry-run]

Examples:
    # Sync all schemas
    python scripts/sync_motherduck_to_local.py

    # Sync only raw and features
    python scripts/sync_motherduck_to_local.py --schemas raw,features

    # Dry run (show what would be synced)
    python scripts/sync_motherduck_to_local.py --dry-run

Architecture:
    1. Connect to LOCAL DuckDB (primary connection)
    2. ATTACH MotherDuck as 'md_source' (hybrid execution enabled)
    3. For each table: CREATE OR REPLACE TABLE local.schema.table AS SELECT * FROM md_source.schema.table
    4. Verify row counts match
    5. Result: Full local copy for fast training I/O
"""
import argparse
import os
import sys
from pathlib import Path

import duckdb

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "cbi_v15.duckdb"

# MotherDuck config
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# All schemas to sync (in order)
ALL_SCHEMAS = [
    "raw",
    "staging",
    "features",
    "training",
    "forecasts",
    "reference",
    "ops",
]


def get_local_connection():
    """
    Connect to local DuckDB (primary connection).
    MotherDuck will be attached to this connection.
    """
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = duckdb.connect(str(LOCAL_DB_PATH))
        print(f"✅ Connected to local DuckDB: {LOCAL_DB_PATH}")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to local DuckDB: {e}")
        sys.exit(1)


def get_motherduck_connection():
    """
    Create a direct connection to MotherDuck (workaround for ATTACH CREATE_SLT issue).
    Returns connection object or None if failed.
    """
    if not MOTHERDUCK_TOKEN:
        return None

    try:
        md_conn = duckdb.connect(
            f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}"
        )
        # Test connection
        md_conn.execute("SELECT 1").fetchone()
        return md_conn
    except Exception as e:
        return None


def attach_motherduck(conn):
    """
    Attach MotherDuck to local connection using hybrid query execution.
    This enables querying MotherDuck data from local DuckDB connection.

    If ATTACH fails (CREATE_SLT RPC error), returns None to signal fallback to direct connection.
    """
    if not MOTHERDUCK_TOKEN:
        print("❌ ERROR: MOTHERDUCK_TOKEN not set in environment")
        print("   Run: export MOTHERDUCK_TOKEN=your_token")
        print("   Or add to .env file")
        sys.exit(1)

    # Try environment variable approach first (DuckDB auto-detects motherduck_token env var)
    motherduck_token_env = os.getenv("motherduck_token") or os.getenv(
        "MOTHERDUCK_TOKEN"
    )

    attach_formats = [
        {
            "name": "env_var_token",
            "query": f"ATTACH 'md:{MOTHERDUCK_DB}' AS md_source",
            "requires_env": True,
        },
        {
            "name": "original",
            "query": f"ATTACH 'md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}' AS md_source",
            "requires_env": False,
        },
    ]

    for fmt in attach_formats:
        # Set environment variable if needed for this format
        if fmt.get("requires_env") and not os.getenv("motherduck_token"):
            # Set lowercase version for DuckDB auto-detection
            os.environ["motherduck_token"] = MOTHERDUCK_TOKEN

        try:
            if fmt["name"] == "original":
                print(
                    f"\n📎 Attaching MotherDuck database '{MOTHERDUCK_DB}' as 'md_source'..."
                )
            elif fmt["name"] == "env_var_token":
                print(f"\n📎 Attaching MotherDuck using environment variable token...")
            else:
                print(f"\n📎 Trying {fmt['name']} format...")

            conn.execute(fmt["query"])

            print("✅ MotherDuck attached successfully")
            print(f"   Hybrid execution enabled: local ↔ cloud")
            return True

        except Exception as e:
            if fmt["name"] == "original":
                print(f"❌ Failed to attach MotherDuck: {e}")
            else:
                print(f"❌ {fmt['name']} format failed: {e}")

            # Continue to next format
            continue

    # All formats failed - return None to signal fallback to direct connection
    print(f"\n⚠️  ATTACH failed (CREATE_SLT RPC error)")
    print(f"   Falling back to direct MotherDuck connection method...")
    return None


def get_tables_in_schema(conn, schema_name, source="md_source", md_conn=None):
    """Get all tables in a schema from specified source (md_source or local)"""
    try:
        if md_conn:
            # Use direct connection
            result = md_conn.execute(
                f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema_name}'
                ORDER BY table_name
            """
            ).fetchall()
        else:
            # Use ATTACH
            result = conn.execute(
                f"""
                SELECT table_name 
                FROM {source}.information_schema.tables 
                WHERE table_schema = '{schema_name}'
                ORDER BY table_name
            """
            ).fetchall()
        return [row[0] for row in result]
    except Exception as e:
        print(f"   ⚠️  Could not list tables in {schema_name}: {e}")
        return []


def get_row_count(conn, schema_name, table_name, source="md_source"):
    """Get row count for a table from specified source"""
    try:
        full_name = f"{source}.{schema_name}.{table_name}"
        result = conn.execute(f"SELECT COUNT(*) FROM {full_name}").fetchone()
        return result[0] if result else 0
    except Exception as e:
        # print(f"   ⚠️  Could not count rows in {full_name}: {e}")
        return -1


def get_row_count_direct(md_conn, schema_name, table_name):
    """Get row count using direct MotherDuck connection"""
    try:
        full_name = f"{schema_name}.{table_name}"
        result = md_conn.execute(f"SELECT COUNT(*) FROM {full_name}").fetchone()
        return result[0] if result else 0
    except Exception as e:
        return -1


def sync_table(conn, schema_name, table_name, dry_run=False, md_conn=None):
    """
    Sync a single table from MotherDuck to local.
    Uses CREATE OR REPLACE TABLE for atomic replacement.

    Args:
        conn: Local DuckDB connection
        schema_name: Schema name
        table_name: Table name
        dry_run: If True, don't actually sync
        md_conn: Direct MotherDuck connection (if ATTACH failed)
    """
    local_full = f"{schema_name}.{table_name}"

    # Get row counts
    if md_conn:
        # Use direct connection
        md_rows = get_row_count_direct(md_conn, schema_name, table_name)
        md_source_desc = "MotherDuck (direct)"
    else:
        # Use ATTACH
        md_rows = get_row_count(conn, schema_name, table_name, source="md_source")
        md_source_desc = "MotherDuck (ATTACH)"

    local_rows = get_row_count(conn, schema_name, table_name, source="memory")

    if md_rows == -1:
        print(f"   ⚠️  Skipping {table_name} (MotherDuck read error)")
        return False

    if md_rows == 0:
        print(f"   ⏭️  Skipping {table_name} (MotherDuck table empty)")
        return True

    # Show sync info
    local_status = f"{local_rows:,}" if local_rows >= 0 else "not exists"
    print(
        f"   📊 {table_name}: {md_source_desc}={md_rows:,} rows → Local={local_status}"
    )

    if dry_run:
        print(f"      🔍 [DRY RUN] Would sync {md_rows:,} rows")
        return True

    try:
        # Atomic replace: CREATE OR REPLACE TABLE
        print(f"      🔄 Syncing {md_rows:,} rows...", end=" ", flush=True)

        if md_conn:
            # Direct connection: query from MotherDuck and insert into local
            data = md_conn.execute(
                f"SELECT * FROM {schema_name}.{table_name}"
            ).fetchdf()
            conn.register("temp_md_data", data)
            conn.execute(
                f"CREATE OR REPLACE TABLE {local_full} AS SELECT * FROM temp_md_data"
            )
            conn.unregister("temp_md_data")
        else:
            # ATTACH method: use hybrid query
            md_full = f"md_source.{schema_name}.{table_name}"
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE {local_full} AS 
                SELECT * FROM {md_full}
            """
            )

        # Verify
        new_count = get_row_count(conn, schema_name, table_name, source="memory")
        if new_count == md_rows:
            print(f"✅ {new_count:,} rows")
            return True
        else:
            print(f"⚠️  Row mismatch: expected {md_rows:,}, got {new_count:,}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def sync_schemas(schemas, dry_run=False):
    """Sync specified schemas from MotherDuck to local"""
    print("\n" + "=" * 80)
    print("SYNC MOTHERDUCK → LOCAL DUCKDB")
    print("=" * 80)

    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made\n")

    # Connect to local DuckDB (primary connection)
    conn = get_local_connection()

    # Try to attach MotherDuck (may fail with CREATE_SLT error)
    md_conn = None
    attach_result = attach_motherduck(conn)

    # If ATTACH failed, use direct connection as fallback
    if attach_result is None:
        md_conn = get_motherduck_connection()
        if md_conn is None:
            print("❌ Failed to connect to MotherDuck via direct connection")
            print("   Both ATTACH and direct connection methods failed")
            sys.exit(1)
        print("✅ Using direct MotherDuck connection (ATTACH unavailable)")

    # Sync each schema
    total_tables = 0
    synced_tables = 0

    for schema in schemas:
        print(f"\n{'─' * 80}")
        print(f"📁 Schema: {schema}")
        print(f"{'─' * 80}")

        # Ensure local schema exists
        try:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        except Exception as e:
            print(f"   ⚠️  Could not create schema {schema}: {e}")
            continue

        # Get tables from MotherDuck
        tables = get_tables_in_schema(conn, schema, source="md_source", md_conn=md_conn)

        if not tables:
            print(f"   📭 No tables in {schema}")
            continue

        print(f"   📋 Found {len(tables)} table(s) in MotherDuck")

        # Sync each table
        for table in tables:
            total_tables += 1
            if sync_table(conn, schema, table, dry_run, md_conn=md_conn):
                synced_tables += 1

    # Summary
    print("\n" + "=" * 80)
    if dry_run:
        print(f"🔍 DRY RUN COMPLETE")
    else:
        print(f"✅ SYNC COMPLETE")
    print(f"📊 Tables synced: {synced_tables}/{total_tables}")

    # Database size
    if not dry_run:
        size_mb = LOCAL_DB_PATH.stat().st_size / (1024 * 1024)
        print(f"💾 Local database size: {size_mb:.1f} MB")

    print("=" * 80 + "\n")

    # Close connections
    conn.close()
    if md_conn:
        md_conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Sync data from MotherDuck to local DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--schemas",
        type=str,
        help=f"Comma-separated list of schemas to sync (default: all). Available: {', '.join(ALL_SCHEMAS)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without making changes",
    )

    args = parser.parse_args()

    # Determine which schemas to sync
    if args.schemas:
        schemas = [s.strip() for s in args.schemas.split(",")]
        # Validate schema names
        invalid = [s for s in schemas if s not in ALL_SCHEMAS]
        if invalid:
            print(f"❌ Invalid schema names: {', '.join(invalid)}")
            print(f"   Valid schemas: {', '.join(ALL_SCHEMAS)}")
            sys.exit(1)
    else:
        schemas = ALL_SCHEMAS

    # Run sync
    sync_schemas(schemas, dry_run=args.dry_run)


if __name__ == "__main__":
    main()














