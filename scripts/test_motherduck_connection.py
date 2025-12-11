#!/usr/bin/env python3
"""
Test MotherDuck Connection
Verifies that MOTHERDUCK_TOKEN is set and can successfully connect to MotherDuck.

Usage:
    python scripts/test_motherduck_connection.py
"""
import os
import sys
from pathlib import Path

import duckdb

# MotherDuck config
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def test_motherduck_connection():
    """Test MotherDuck connection and token validity"""
    print("=" * 80)
    print("MOTHERDUCK CONNECTION TEST")
    print("=" * 80)
    print()

    # 1. Check token is set
    if not MOTHERDUCK_TOKEN:
        print("❌ MOTHERDUCK_TOKEN not set in environment")
        print()
        print("💡 To set the token:")
        print("   export MOTHERDUCK_TOKEN=your_token")
        print("   # Or add to .env file")
        print()
        sys.exit(1)

    print(f"✅ MOTHERDUCK_TOKEN is set (length: {len(MOTHERDUCK_TOKEN)} chars)")
    print(f"📊 Database: {MOTHERDUCK_DB}")
    print()

    # 2. Test connection
    print("🔌 Testing connection...")
    try:
        conn = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
        print("✅ Connection established")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("💡 Troubleshooting:")
        print("   1. Verify token is valid at https://app.motherduck.com/")
        print("   2. Check database name is correct (default: cbi_v15)")
        print("   3. Ensure token has read/write permissions")
        sys.exit(1)

    # 3. Test query
    print("🔍 Testing query execution...")
    try:
        result = conn.execute("SELECT 1 as test").fetchone()
        if result and result[0] == 1:
            print("✅ Query execution successful")
        else:
            print("⚠️  Query returned unexpected result")
    except Exception as e:
        print(f"❌ Query execution failed: {e}")
        conn.close()
        sys.exit(1)

    # 4. List schemas
    print()
    print("📁 Listing available schemas...")
    try:
        schemas = conn.execute(
            """
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_temp')
            ORDER BY schema_name
        """
        ).fetchall()

        if schemas:
            print(f"✅ Found {len(schemas)} schema(s):")
            for schema in schemas:
                print(f"   - {schema[0]}")
        else:
            print("⚠️  No schemas found (database may be empty)")
    except Exception as e:
        print(f"⚠️  Could not list schemas: {e}")

    # 5. Test ATTACH mode (for sync script compatibility)
    print()
    print("🔗 Testing ATTACH mode (for sync script)...")
    try:
        # Create a temporary local connection
        local_conn = duckdb.connect(":memory:")
        local_conn.execute(
            f"ATTACH 'md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}' AS md_test"
        )
        test_result = local_conn.execute(
            "SELECT 1 FROM md_test.information_schema.tables LIMIT 1"
        ).fetchone()
        local_conn.close()
        print("✅ ATTACH mode works (compatible with sync script)")
    except Exception as e:
        print(f"⚠️  ATTACH mode test failed: {e}")
        print("   (This may affect sync_motherduck_to_local.py)")

    # Summary
    print()
    print("=" * 80)
    print("✅ MOTHERDUCK CONNECTION TEST PASSED")
    print("=" * 80)
    print()
    print("Connection details:")
    print(f"  Database: {MOTHERDUCK_DB}")
    print(
        f"  Token: {'*' * 20}...{MOTHERDUCK_TOKEN[-4:] if len(MOTHERDUCK_TOKEN) > 4 else '****'}"
    )
    print()
    print("You can now:")
    print("  - Run: python scripts/sync_motherduck_to_local.py")
    print("  - Query MotherDuck directly from Python scripts")
    print()

    conn.close()


if __name__ == "__main__":
    test_motherduck_connection()


