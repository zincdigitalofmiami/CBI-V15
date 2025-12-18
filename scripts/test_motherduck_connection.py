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

def _load_dotenv_file(path: Path) -> None:
    """
    Lightweight dotenv loader (no deps).
    - Only sets keys that are not already present in the environment.
    - Supports KEY=VALUE with optional surrounding quotes.
    - Ignores blank lines and comments starting with '#'.
    """
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def _load_local_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv_file(project_root / ".env")
    _load_dotenv_file(project_root / ".env.local")


_load_local_env()

# MotherDuck config (read from environment after dotenv load)
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

def _iter_tokens():
    candidates = [
        ("MOTHERDUCK_TOKEN", os.getenv("MOTHERDUCK_TOKEN")),
        ("motherduck_storage_MOTHERDUCK_TOKEN", os.getenv("motherduck_storage_MOTHERDUCK_TOKEN")),
        ("MOTHERDUCK_READ_SCALING_TOKEN", os.getenv("MOTHERDUCK_READ_SCALING_TOKEN")),
        ("motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN", os.getenv("motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN")),
    ]
    for name, value in candidates:
        if not value:
            continue
        token = value.strip().strip('"').strip("'")
        # Basic JWT shape check: 3 sections separated by dots
        if token.count(".") != 2:
            continue
        yield name, token


def test_motherduck_connection():
    """Test MotherDuck connection and token validity"""
    print("=" * 80)
    print("MOTHERDUCK CONNECTION TEST")
    print("=" * 80)
    print()

    # 1. Check token is set
    token_candidates = list(_iter_tokens())
    if not token_candidates:
        print("❌ No MotherDuck token found in environment (.env/.env.local)")
        print()
        print("💡 To set the token:")
        print("   export MOTHERDUCK_TOKEN=your_token  # or motherduck_storage_MOTHERDUCK_TOKEN")
        print("   # Or add to .env/.env.local")
        print()
        sys.exit(1)

    print(f"✅ Found {len(token_candidates)} token candidate(s)")
    print(f"📊 Database: {MOTHERDUCK_DB}")
    print()

    # 2. Test connection
    print("🔌 Testing connection...")
    try:
        conn = None
        token_source = None
        token_used = None
        last_error = None

        for name, token in token_candidates:
            try:
                conn = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={token}")
                conn.execute("SELECT 1").fetchone()
                token_source = name
                token_used = token
                break
            except Exception as e:
                last_error = e
                conn = None

        if conn is None:
            raise last_error or RuntimeError("No token candidates worked")

        print("✅ Connection established")
        print(f"🔑 Token source: {token_source}")
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

    # 4b. Storage info (helps reconcile MotherDuck UI storage numbers)
    print()
    print("💾 Storage info (per owner)...")
    try:
        rows = conn.execute(
            """
            SELECT
              username,
              active_bytes,
              historical_bytes,
              kept_for_cloned_bytes,
              failsafe_bytes,
              (active_bytes + historical_bytes + kept_for_cloned_bytes + failsafe_bytes) AS total_bytes
            FROM md_information_schema.main.storage_info
            WHERE database_name = ?
            ORDER BY total_bytes DESC
        """,
            [MOTHERDUCK_DB],
        ).fetchall()

        if rows:
            for username, active_b, hist_b, kept_b, failsafe_b, total_b in rows:
                print(
                    f"   - {username}: total={total_b:,} (active={active_b:,} hist={hist_b:,} failsafe={failsafe_b:,})"
                )
        else:
            print("   ⚠️  No storage info rows found")
    except Exception as e:
        print(f"⚠️  Could not read storage info: {e}")

    # 5. Test ATTACH mode (for sync script compatibility)
    print()
    print("🔗 Testing ATTACH mode (for sync script)...")
    try:
        # Create a temporary local connection
        local_conn = duckdb.connect(":memory:")
        local_conn.execute(
            f"ATTACH 'md:{MOTHERDUCK_DB}?motherduck_token={token_used}' AS md_test"
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
        f"  Token: {'*' * 20}...{token_used[-4:] if token_used and len(token_used) > 4 else '****'}"
    )
    print()
    print("You can now:")
    print("  - Run: python scripts/sync_motherduck_to_local.py")
    print("  - Query MotherDuck directly from Python scripts")
    print()

    conn.close()


if __name__ == "__main__":
    test_motherduck_connection()













