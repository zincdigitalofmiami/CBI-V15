#!/usr/bin/env python3
"""
DIAGNOSTIC REPORT ONLY - NO REPAIRS

Runs the three diagnostic queries requested:
1. Check MotherDuck table columns (should be 38)
2. Check macro output columns (should be 39)
3. Find the difference between them

This script only reports findings - it does NOT make any changes.
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
    print("TECHNICAL INDICATORS DIAGNOSTIC REPORT")
    print("REPORT ONLY - NO CHANGES WILL BE MADE")
    print("=" * 80)
    print()

    # ========================================================================
    # QUERY 1: Check MotherDuck table columns (Expected: 38)
    # ========================================================================
    print("QUERY 1: Check MotherDuck table column count")
    print("-" * 80)
    print("SQL:")
    print(
        """
SELECT COUNT(*) as column_count
FROM information_schema.columns
WHERE table_schema = 'features' 
  AND table_name = 'technical_indicators_all_symbols';
-- Expected: 38
    """.strip()
    )
    print()

    result = conn.execute(
        """
        SELECT COUNT(*) as column_count
        FROM information_schema.columns
        WHERE table_schema = 'features' 
          AND table_name = 'technical_indicators_all_symbols'
    """
    ).fetchone()

    table_column_count = result[0] if result else 0
    print(f"Result: {table_column_count} columns")
    print(f"Expected: 38 columns")
    if table_column_count == 38:
        print("✅ PASS: Column count matches expected")
    else:
        print(f"❌ FAIL: Column count mismatch (got {table_column_count}, expected 38)")
    print()
    print()

    # ========================================================================
    # QUERY 2: Check macro output columns (Expected: 39)
    # ========================================================================
    print("QUERY 2: Check macro output column count")
    print("-" * 80)
    print("SQL:")
    print(
        """
-- This requires macros loaded first
SELECT COUNT(*) as column_count
FROM (SELECT * FROM calc_all_technical_indicators('ZL') LIMIT 0);
-- Expected: 39
    """.strip()
    )
    print()

    # Load required macros
    print("Loading macros...")
    macro_files = [
        "database/macros/features.sql",
        "database/macros/technical_indicators_all_symbols.sql",
    ]

    for macro_file in macro_files:
        macro_path = Path(__file__).parent.parent.parent / macro_file
        if not macro_path.exists():
            print(f"❌ Macro file not found: {macro_path}")
            sys.exit(1)

        with open(macro_path, "r") as f:
            macro_sql = f.read()
            conn.execute(macro_sql)

    print("✅ Macros loaded")
    print()

    # Count macro output columns
    macro_result = conn.execute(
        """
        SELECT COUNT(*) as column_count
        FROM (SELECT * FROM calc_all_technical_indicators('ZL') LIMIT 0)
    """
    ).fetchone()

    macro_column_count = macro_result[0] if macro_result else 0
    print(f"Result: {macro_column_count} columns")
    print(f"Expected: 39 columns")
    if macro_column_count == 39:
        print("✅ PASS: Column count matches expected")
    else:
        print(f"❌ FAIL: Column count mismatch (got {macro_column_count}, expected 39)")
    print()
    print()

    # ========================================================================
    # QUERY 3: Find the difference
    # ========================================================================
    print("QUERY 3: Find the difference between table and macro")
    print("-" * 80)
    print("SQL Part 1 - Get column names from table:")
    print(
        """
SELECT column_name 
FROM information_schema.columns
WHERE table_schema = 'features' 
  AND table_name = 'technical_indicators_all_symbols'
ORDER BY ordinal_position;
    """.strip()
    )
    print()

    # Get table column names
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

    print("SQL Part 2 - Compare with macro output:")
    print(
        """
DESCRIBE SELECT * FROM calc_all_technical_indicators('ZL') LIMIT 0;
    """.strip()
    )
    print()

    # Get macro column names
    macro_columns = conn.execute(
        """
        DESCRIBE SELECT * FROM calc_all_technical_indicators('ZL') LIMIT 0
    """
    ).fetchall()

    macro_column_names = [col[0] for col in macro_columns]
    print(f"Macro columns ({len(macro_column_names)}):")
    for i, col in enumerate(macro_column_names, 1):
        print(f"  {i:2d}. {col}")
    print()

    # ========================================================================
    # DIFFERENCE ANALYSIS
    # ========================================================================
    print("=" * 80)
    print("DIFFERENCE ANALYSIS")
    print("=" * 80)
    print()

    table_set = set(table_column_names)
    macro_set = set(macro_column_names)

    # Columns in macro but not in table
    missing_in_table = macro_set - table_set
    print(f"Columns in MACRO but MISSING from TABLE: {len(missing_in_table)}")
    if missing_in_table:
        for col in sorted(missing_in_table):
            print(f"  ❌ {col}")
    else:
        print("  ✅ None")
    print()

    # Columns in table but not in macro
    missing_in_macro = table_set - macro_set
    print(f"Columns in TABLE but MISSING from MACRO: {len(missing_in_macro)}")
    if missing_in_macro:
        for col in sorted(missing_in_macro):
            print(f"  ❌ {col}")
    else:
        print("  ✅ None")
    print()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print(f"Table columns:           {len(table_column_names)}")
    print(f"Macro columns:           {len(macro_column_names)}")
    print(f"Columns missing in table: {len(missing_in_table)}")
    print(f"Columns missing in macro: {len(missing_in_macro)}")
    print()

    # Overall assessment
    if len(table_column_names) == 38 and len(macro_column_names) == 39:
        print("✅ Column counts match expected values (38 table, 39 macro)")
    else:
        print(
            f"❌ Column counts do NOT match expected values (expected 38 table, 39 macro)"
        )
    print()

    if missing_in_table or missing_in_macro:
        print("⚠️  MISMATCH DETECTED")
        if missing_in_macro == {"updated_at"}:
            print(
                "ℹ️  Note: 'updated_at' is a system-managed timestamp column in the table."
            )
            print("   This is expected and should NOT be included in macro output.")
            print("   This is CORRECT behavior.")
        else:
            print("   This requires investigation and potential repair.")
        return 1
    else:
        print("✅ No mismatch - Columns are perfectly aligned")
        return 0

    conn.close()


if __name__ == "__main__":
    sys.exit(main())


