#!/usr/bin/env python3
"""
Test the fixed build_all_features.py with a single symbol
to verify the updated_at column fix works
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

    conn = duckdb.connect(f"md:cbi_v15?motherduck_token={motherduck_token}")

    print("=" * 80)
    print("TESTING FIXED build_all_features.py")
    print("Testing with single symbol: ZL")
    print("=" * 80)
    print()

    # Load macros
    print("Step 1: Loading macros...")
    print("-" * 80)
    macro_files = [
        "database/macros/features.sql",
        "database/macros/technical_indicators_all_symbols.sql",
    ]

    for macro_file in macro_files:
        macro_path = Path(__file__).parent.parent.parent / macro_file
        if not macro_path.exists():
            print(f"❌ Macro file not found: {macro_path}")
            sys.exit(1)

        print(f"  Loading {macro_file}...")
        with open(macro_path, "r") as f:
            conn.execute(f.read())

    print("✅ Macros loaded")
    print()

    # Clear existing ZL data
    print("Step 2: Clearing existing ZL data...")
    print("-" * 80)
    conn.execute(
        "DELETE FROM features.technical_indicators_all_symbols WHERE symbol = 'ZL'"
    )
    print("✅ Cleared")
    print()

    # Test the fixed INSERT statement
    print("Step 3: Testing fixed INSERT statement...")
    print("-" * 80)
    print("SQL:")
    print(
        """
INSERT OR REPLACE INTO features.technical_indicators_all_symbols
SELECT 
    *,
    CURRENT_TIMESTAMP as updated_at
FROM calc_all_technical_indicators('ZL')
    """.strip()
    )
    print()

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO features.technical_indicators_all_symbols
            SELECT 
                *,
                CURRENT_TIMESTAMP as updated_at
            FROM calc_all_technical_indicators('ZL')
        """
        )
        print("✅ INSERT SUCCEEDED!")
        print()
    except Exception as e:
        print(f"❌ INSERT FAILED: {e}")
        sys.exit(1)

    # Verify results
    print("Step 4: Verifying results...")
    print("-" * 80)

    # Check row count
    result = conn.execute(
        """
        SELECT 
            COUNT(*) as row_count,
            MIN(as_of_date) as min_date,
            MAX(as_of_date) as max_date,
            MIN(updated_at) as min_ts,
            MAX(updated_at) as max_ts
        FROM features.technical_indicators_all_symbols 
        WHERE symbol = 'ZL'
    """
    ).fetchone()

    print(f"Rows inserted:     {result[0]:,}")
    print(f"Date range:        {result[1]} to {result[2]}")
    print(f"Updated timestamp: {result[3]}")
    print()

    # Check for NULL values in key columns
    null_check = conn.execute(
        """
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
            SUM(CASE WHEN rsi_14 IS NULL THEN 1 ELSE 0 END) as null_rsi,
            SUM(CASE WHEN updated_at IS NULL THEN 1 ELSE 0 END) as null_updated_at
        FROM features.technical_indicators_all_symbols 
        WHERE symbol = 'ZL'
    """
    ).fetchone()

    print(f"NULL checks:")
    print(f"  Total rows:        {null_check[0]:,}")
    print(f"  NULL close:        {null_check[1]}")
    print(f"  NULL rsi_14:       {null_check[2]}")
    print(f"  NULL updated_at:   {null_check[3]}")
    print()

    if null_check[3] > 0:
        print("❌ FAIL: updated_at has NULL values")
        sys.exit(1)
    else:
        print("✅ PASS: All updated_at values populated")

    # Sample data
    print()
    print("Step 5: Sample data (latest 3 rows)...")
    print("-" * 80)
    sample = conn.execute(
        """
        SELECT 
            as_of_date,
            symbol,
            close,
            rsi_14,
            macd_histogram,
            updated_at
        FROM features.technical_indicators_all_symbols 
        WHERE symbol = 'ZL'
        ORDER BY as_of_date DESC
        LIMIT 3
    """
    ).fetchdf()

    print(sample.to_string(index=False))
    print()

    print("=" * 80)
    print("✅ TEST PASSED - Fix is working correctly!")
    print("=" * 80)
    print()
    print("Next step: Run full build_all_features.py for all symbols")

    conn.close()


if __name__ == "__main__":
    main()


