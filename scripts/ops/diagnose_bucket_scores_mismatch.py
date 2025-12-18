#!/usr/bin/env python3
"""
Diagnostic for bucket_scores schema mismatch
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import duckdb
from dotenv import load_dotenv

load_dotenv()


def main():
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        print("❌ MOTHERDUCK_TOKEN not found")
        sys.exit(1)

    con = duckdb.connect(f"md:cbi_v15?motherduck_token={motherduck_token}")

    print("=" * 80)
    print("🔍 BUCKET SCORES SCHEMA MISMATCH DIAGNOSTIC")
    print("=" * 80)
    print()

    # Load macros
    print("Loading macros...")
    macro_path = (
        Path(__file__).parent.parent.parent / "database/macros/big8_bucket_features.sql"
    )
    with open(macro_path, "r") as f:
        con.execute(f.read())
    print("✅ Macros loaded")
    print()

    # Check table schema
    print("📋 Table columns:")
    print("-" * 80)
    table_cols = con.execute(
        """
        SELECT column_name 
        FROM information_schema.columns
        WHERE table_schema = 'features' 
          AND table_name = 'bucket_scores'
        ORDER BY ordinal_position
    """
    ).fetchall()
    print(f"Count: {len(table_cols)}")
    for i, col in enumerate(table_cols, 1):
        print(f"  {i:2d}. {col[0]}")
    print()

    # Check macro output
    print("📋 Macro output columns:")
    print("-" * 80)
    con.execute(
        "CREATE TEMP TABLE macro_test AS SELECT * FROM calc_all_bucket_scores() LIMIT 0"
    )
    macro_cols = con.execute(
        """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'macro_test'
        ORDER BY ordinal_position
    """
    ).fetchall()
    print(f"Count: {len(macro_cols)}")
    for i, col in enumerate(macro_cols, 1):
        print(f"  {i:2d}. {col[0]}")
    print()

    # Find differences
    print("🔍 Differences:")
    print("-" * 80)
    table_set = set(c[0] for c in table_cols)
    macro_set = set(c[0] for c in macro_cols)

    missing_in_macro = table_set - macro_set
    missing_in_table = macro_set - table_set

    print(f"In table but NOT in macro ({len(missing_in_macro)}):")
    for col in sorted(missing_in_macro):
        print(f"  ❌ {col}")
    print()

    print(f"In macro but NOT in table ({len(missing_in_table)}):")
    for col in sorted(missing_in_table):
        print(f"  ❌ {col}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Table columns:  {len(table_cols)}")
    print(f"Macro columns:  {len(macro_cols)}")
    print(f"Missing in macro: {len(missing_in_macro)}")
    print(f"Missing in table: {len(missing_in_table)}")
    print()

    # Determine fix
    if missing_in_macro == {"updated_at"}:
        print("✅ SCENARIO A: Only missing updated_at")
        print("   Fix: Add CURRENT_TIMESTAMP as updated_at")
    elif missing_in_macro == {"created_at", "updated_at"}:
        print("✅ SCENARIO B: Missing created_at and updated_at")
        print("   Fix: Add both timestamps")
    else:
        print("⚠️  SCENARIO C: Schema drift detected")
        print("   Fix: Manual intervention needed")

    con.close()


if __name__ == "__main__":
    main()



