#!/usr/bin/env python3
"""
Test if build_symbol_features macro works for ZL
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
    print("TESTING build_symbol_features('ZL')")
    print("=" * 80)
    print()

    # Load all macros
    print("Loading macros...")
    macro_files = [
        "database/macros/features.sql",
        "database/macros/technical_indicators_all_symbols.sql",
        "database/macros/cross_asset_features.sql",
        "database/macros/big8_bucket_features.sql",
        "database/macros/master_feature_matrix.sql",
    ]

    for macro_file in macro_files:
        macro_path = Path(__file__).parent.parent.parent / macro_file
        if not macro_path.exists():
            print(f"  ⚠️  {macro_file} not found, skipping")
            continue

        print(f"  Loading {macro_file}...")
        with open(macro_path, "r") as f:
            try:
                con.execute(f.read())
            except Exception as e:
                print(f"    ❌ Error loading: {e}")
                return 1

    print("✅ Macros loaded")
    print()

    # Test the macro
    print("Testing build_symbol_features('ZL')...")
    print("-" * 80)

    try:
        # Get column count
        result = con.execute(
            """
            SELECT COUNT(*) as col_count
            FROM (SELECT * FROM build_symbol_features('ZL') LIMIT 0)
        """
        ).fetchone()

        col_count = result[0] if result else 0
        print(f"✅ Macro executed successfully")
        print(f"   Columns returned: {col_count}")
        print()

        # Get column names
        print("Column names:")
        cols = con.execute(
            """
            DESCRIBE SELECT * FROM build_symbol_features('ZL') LIMIT 0
        """
        ).fetchall()

        for i, col in enumerate(cols[:20], 1):  # Show first 20
            print(f"  {i:2d}. {col[0]}")
        if len(cols) > 20:
            print(f"  ... and {len(cols) - 20} more columns")
        print()

        # Try to get actual data
        print("Testing with real data (LIMIT 5)...")
        data = con.execute(
            """
            SELECT * FROM build_symbol_features('ZL') 
            ORDER BY as_of_date DESC
            LIMIT 5
        """
        ).fetchdf()

        print(f"✅ Retrieved {len(data)} rows")
        print(
            f"   Date range: {data['as_of_date'].min()} to {data['as_of_date'].max()}"
        )
        print()

        # Check table schema
        print("Checking daily_ml_matrix_zl table schema...")
        table_cols = con.execute(
            """
            SELECT COUNT(*) as col_count
            FROM information_schema.columns
            WHERE table_schema = 'features' 
              AND table_name = 'daily_ml_matrix_zl'
        """
        ).fetchone()

        table_col_count = table_cols[0] if table_cols else 0
        print(f"   Table has: {table_col_count} columns")
        print(f"   Macro has: {col_count} columns")

        if table_col_count == col_count + 1:
            print("   ✅ Difference is 1 (likely updated_at)")
            print("   Fix: Add CURRENT_TIMESTAMP as updated_at to INSERT")
        elif table_col_count == col_count:
            print("   ✅ Perfect match!")
        else:
            print(
                f"   ⚠️  Mismatch: {abs(table_col_count - col_count)} columns difference"
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    print()
    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())



