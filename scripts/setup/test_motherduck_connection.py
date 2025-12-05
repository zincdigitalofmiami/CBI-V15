#!/usr/bin/env python3
"""
Setup MotherDuck Service Token
Validates and tests the MotherDuck connection with your service token.
"""

import os
import sys
from pathlib import Path


def main():
    # Check for token in environment
    token = os.getenv("MOTHERDUCK_TOKEN")

    if not token:
        print("❌ MOTHERDUCK_TOKEN not found in environment")
        print("\n📝 To set it up:")
        print("   1. Create .env file in project root:")
        print("      echo 'MOTHERDUCK_TOKEN=your_token_here' > .env")
        print("   2. Source it:")
        print("      export $(cat .env | xargs)")
        print("   3. Run this script again")
        sys.exit(1)

    print("✅ MOTHERDUCK_TOKEN found")
    print(f"   Token length: {len(token)} chars")
    print(f"   Prefix: {token[:10]}...")

    # Test connection
    try:
        import duckdb

        print("\n🔌 Testing connection to md:cbi_v15...")
        con = duckdb.connect(f"md:cbi_v15?motherduck_token={token}")

        print("✅ Connected successfully!")

        # List schemas
        print("\n📊 Available Schemas:")
        schemas = con.execute("SHOW SCHEMAS").fetchall()
        for schema in schemas:
            if schema[0] not in ["information_schema", "pg_catalog"]:
                print(f"   ✓ {schema[0]}")

        # List tables in key schemas
        for schema_name in ["raw", "staging", "features", "training", "forecasts"]:
            try:
                tables = con.execute(f"SHOW TABLES FROM {schema_name}").fetchall()
                if tables:
                    print(f"\n📁 Tables in {schema_name}:")
                    for table in tables:
                        row_count = con.execute(
                            f"SELECT COUNT(*) FROM {schema_name}.{table[0]}"
                        ).fetchone()[0]
                        print(f"   ✓ {table[0]} ({row_count:,} rows)")
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
