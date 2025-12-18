#!/usr/bin/env python3
"""
Setup MotherDuck Service Token
Validates and tests the MotherDuck connection with your service token.
"""

import os
import sys
from pathlib import Path


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
    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv_file(project_root / ".env")
    _load_dotenv_file(project_root / ".env.local")


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
        if token.count(".") != 2:
            continue
        yield name, token


def main():
    _load_local_env()

    # Check for token in environment
    token_candidates = list(_iter_tokens())
    if not token_candidates:
        print("❌ No MotherDuck token found in environment (.env/.env.local)")
        print("\n📝 To set it up:")
        print("   export MOTHERDUCK_TOKEN=your_token_here")
        print("   # or: export motherduck_storage_MOTHERDUCK_TOKEN=your_token_here")
        sys.exit(1)

    print(f"✅ Found {len(token_candidates)} token candidate(s)")
    for name, token in token_candidates:
        print(f"   ✓ {name} (len={len(token)})")

    # Test connection
    try:
        import duckdb

        print("\n🔌 Testing connection to md:cbi_v15...")
        con = None
        last_error = None
        for _, token in token_candidates:
            try:
                con = duckdb.connect(f"md:cbi_v15?motherduck_token={token}")
                con.execute("SELECT 1").fetchone()
                break
            except Exception as e:
                last_error = e
                con = None
        if con is None:
            raise last_error or RuntimeError("No token candidates worked")

        print("✅ Connected successfully!")

        # List schemas
        print("\n📊 Available Schemas:")
        schemas = con.execute(
            """
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema','pg_catalog')
            ORDER BY schema_name
            """
        ).fetchall()
        for (schema_name,) in schemas:
            print(f"   ✓ {schema_name}")

        # List tables in key schemas
        for schema_name in ["raw", "staging", "features", "training", "forecasts"]:
            try:
                tables = con.execute(
                    f"""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = '{schema_name}'
                      AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                ).fetchall()
                if tables:
                    print(f"\n📁 Tables in {schema_name}:")
                    for (table_name,) in tables:
                        row_count = con.execute(
                            f'SELECT COUNT(*)::BIGINT FROM "{schema_name}"."{table_name}"'
                        ).fetchone()[0]
                        print(f"   ✓ {table_name} ({row_count:,} rows)")
            except:
                pass

        con.close()
        print("\n✅ MotherDuck service account is ready!")

    except ImportError:
        print("❌ DuckDB not installed")
        print("   Run: pip install duckdb")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print(
            "   1. Verify token is valid at https://app.motherduck.com/settings/tokens"
        )
        print("   2. Check that cbi_v15 database exists")
        print("   3. Ensure service account has read/write access")
        sys.exit(1)


if __name__ == "__main__":
    main()
