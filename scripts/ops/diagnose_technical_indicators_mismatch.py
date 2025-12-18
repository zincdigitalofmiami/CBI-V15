#!/usr/bin/env python3
"""
Diagnostic script to identify column mismatch between:
1. features.technical_indicators_all_symbols table (MotherDuck)
2. calc_all_technical_indicators macro output

Expected: Table should have 38 columns, macro should output 39 columns
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import duckdb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        print("❌ MOTHERDUCK_TOKEN not found in environment")
        sys.exit(1)

    # Connect to MotherDuck
    conn = duckdb.connect(f"md:cbi_v15?motherduck_token={motherduck_token}")

    print("=" * 80)
    print("TECHNICAL INDICATORS COLUMN MISMATCH DIAGNOSTIC")
    print("=" * 80)
    print()

    # Query 1: Check MotherDuck table columns
    print("Query 1: MotherDuck table column count")
    print("-" * 80)
    result = conn.execute(
        """
        SELECT COUNT(*) as column_count
        FROM information_schema.columns
        WHERE table_schema = 'features' 
          AND table_name = 'technical_indicators_all_symbols'
    """
    ).fetchone()
    table_column_count = result[0] if result else 0
    print(f"✅ Table column count: {table_column_count}")
    print()

    # Query 2: Get table column names
    print("Query 2: MotherDuck table column names")
    print("-" * 80)
    table_columns = conn.execute(
        """
        SELECT column_name 
        FROM information_schema.columns
        WHERE table_schema = 'features' 
          AND table_name = 'technical_indicators_all_symbols'
        ORDER BY ordinal_position
    """
    ).fetchall()

    table_column_names = [col[0] for col in table_columns]
    print(f"Table columns ({len(table_column_names)}):")
    for i, col in enumerate(table_column_names, 1):
        print(f"  {i:2d}. {col}")
    print()

    # Query 3: Load macros and check macro output
    print("Query 3: Loading macros and checking macro output")
    print("-" * 80)

    # Load required macros
    macro_files = [
        "database/macros/features.sql",
        "database/macros/technical_indicators_all_symbols.sql",
    ]

    for macro_file in macro_files:
        macro_path = Path(__file__).parent.parent.parent / macro_file
        if not macro_path.exists():
            print(f"❌ Macro file not found: {macro_path}")
            sys.exit(1)

        print(f"Loading: {macro_file}")
        with open(macro_path, "r") as f:
            macro_sql = f.read()
            conn.execute(macro_sql)

    print("✅ Macros loaded successfully")
    print()

    # Query 4: Check macro output columns
    print("Query 4: Macro output column count and names")
    print("-" * 80)

    # Use DESCRIBE to get column info from macro output
    macro_columns = conn.execute(
        """
        DESCRIBE SELECT * FROM calc_all_technical_indicators('ZL') LIMIT 0
    """
    ).fetchall()

    macro_column_names = [col[0] for col in macro_columns]
    print(f"✅ Macro output column count: {len(macro_column_names)}")
    print(f"Macro columns ({len(macro_column_names)}):")
    for i, col in enumerate(macro_column_names, 1):
        print(f"  {i:2d}. {col}")
    print()

    # Query 5: Find the difference
    print("Query 5: Column Differences")
    print("-" * 80)

    table_set = set(table_column_names)
    macro_set = set(macro_column_names)

    # Columns in macro but not in table
    missing_in_table = macro_set - table_set
    if missing_in_table:
        print(f"❌ Columns in MACRO but MISSING from TABLE ({len(missing_in_table)}):")
        for col in sorted(missing_in_table):
            print(f"  - {col}")
    else:
        print("✅ No columns missing from table")
    print()

    # Columns in table but not in macro
    missing_in_macro = table_set - macro_set
    if missing_in_macro:
        print(f"❌ Columns in TABLE but MISSING from MACRO ({len(missing_in_macro)}):")
        for col in sorted(missing_in_macro):
            print(f"  - {col}")
    else:
        print("✅ No columns missing from macro")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Table columns:      {len(table_column_names)}")
    print(f"Macro columns:      {len(macro_column_names)}")
    print(f"Missing from table: {len(missing_in_table)}")
    print(f"Missing from macro: {len(missing_in_macro)}")
    print()

    if missing_in_table or missing_in_macro:
        print("⚠️  MISMATCH DETECTED - Repair needed")
        return 1
    else:
        print("✅ No mismatch - Columns are aligned")
        return 0

    conn.close()


if __name__ == "__main__":
    sys.exit(main())



