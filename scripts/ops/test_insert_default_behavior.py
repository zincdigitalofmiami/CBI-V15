#!/usr/bin/env python3
"""
Test if DuckDB DEFAULT CURRENT_TIMESTAMP works with INSERT OR REPLACE
when columns are explicitly listed (excluding updated_at)
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
    print("TESTING INSERT DEFAULT BEHAVIOR WITH updated_at")
    print("=" * 80)
    print()

    # Load macros
    print("Loading macros...")
    macro_files = [
        Path(__file__).parent.parent.parent / "database/macros/features.sql",
        Path(__file__).parent.parent.parent
        / "database/macros/technical_indicators_all_symbols.sql",
    ]

    for macro_file in macro_files:
        if macro_file.exists():
            with open(macro_file, "r") as f:
                conn.execute(f.read())

    print("✅ Macros loaded")
    print()

    # Test 1: Get column list from macro (38 columns)
    print("Test 1: Get column names from macro output")
    print("-" * 80)
    macro_cols = (
        conn.execute(
            """
        SELECT column_name 
        FROM (DESCRIBE SELECT * FROM calc_all_technical_indicators('ZL'))
    """
        )
        .fetchdf()["column_name"]
        .tolist()
    )

    print(f"Macro returns {len(macro_cols)} columns:")
    print(", ".join(macro_cols[:5]) + ", ...")
    print()

    # Test 2: Try INSERT with explicit column list (should let DEFAULT handle updated_at)
    print("Test 2: INSERT with explicit column list (excluding updated_at)")
    print("-" * 80)

    col_list = ", ".join(macro_cols)

    try:
        # Delete test data first
        conn.execute(
            "DELETE FROM features.technical_indicators_all_symbols WHERE symbol = 'ZL'"
        )

        # Try INSERT with explicit columns
        conn.execute(
            f"""
            INSERT INTO features.technical_indicators_all_symbols
                ({col_list})
            SELECT {col_list}
            FROM calc_all_technical_indicators('ZL')
            LIMIT 10
        """
        )

        print("✅ INSERT with explicit columns SUCCEEDED")
        print()

        # Check if updated_at was populated
        result = conn.execute(
            """
            SELECT COUNT(*) as cnt, MIN(updated_at) as min_ts, MAX(updated_at) as max_ts
            FROM features.technical_indicators_all_symbols 
            WHERE symbol = 'ZL'
        """
        ).fetchone()

        print(f"Rows inserted: {result[0]}")
        print(f"Min updated_at: {result[1]}")
        print(f"Max updated_at: {result[2]}")
        print()

        if result[1] is not None:
            print("✅ DEFAULT CURRENT_TIMESTAMP worked!")
        else:
            print("❌ updated_at is NULL - DEFAULT didn't work")

    except Exception as e:
        print(f"❌ INSERT FAILED: {e}")
        print()

    # Test 3: Try with SELECT *, CURRENT_TIMESTAMP
    print("Test 3: INSERT with SELECT *, CURRENT_TIMESTAMP as updated_at")
    print("-" * 80)

    try:
        # Delete test data first
        conn.execute(
            "DELETE FROM features.technical_indicators_all_symbols WHERE symbol = 'ZL'"
        )

        # Try INSERT with added updated_at
        conn.execute(
            """
            INSERT INTO features.technical_indicators_all_symbols
            SELECT 
                *,
                CURRENT_TIMESTAMP as updated_at
            FROM calc_all_technical_indicators('ZL')
            LIMIT 10
        """
        )

        print("✅ INSERT with added updated_at SUCCEEDED")
        print()

        # Check result
        result = conn.execute(
            """
            SELECT COUNT(*) as cnt, MIN(updated_at) as min_ts, MAX(updated_at) as max_ts
            FROM features.technical_indicators_all_symbols 
            WHERE symbol = 'ZL'
        """
        ).fetchone()

        print(f"Rows inserted: {result[0]}")
        print(f"Min updated_at: {result[1]}")
        print(f"Max updated_at: {result[2]}")
        print()

    except Exception as e:
        print(f"❌ INSERT FAILED: {e}")

    conn.close()

    print("=" * 80)
    print("RECOMMENDATION:")
    print("Use: SELECT *, CURRENT_TIMESTAMP as updated_at")
    print("This is the simplest and most explicit approach.")
    print("=" * 80)


if __name__ == "__main__":
    main()



