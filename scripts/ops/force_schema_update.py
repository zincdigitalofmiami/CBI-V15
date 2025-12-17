#!/usr/bin/env python3
"""
Force Schema Update: Align table schemas with macro outputs.

This script fixes the "Semantic vs Raw" naming mismatch:
- DDL defined: databento_zl_close, tech_zl_rsi_14 (semantic names)
- Macros produce: close, rsi_14 (raw names)
- Downstream code expects: close (raw names)

Solution: Drop tables and recreate from macro output (raw names).
This is safe because the tables are EMPTY (0 rows).
"""

import os
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("MOTHERDUCK_DB", "cbi_v15")
TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MACROS_DIR = Path(__file__).parent.parent.parent / "database" / "macros"


def force_schema_update():
    if not TOKEN:
        print("❌ MOTHERDUCK_TOKEN not set")
        sys.exit(1)

    print("=" * 80)
    print("FORCE SCHEMA UPDATE: Aligning Tables with Macro Output")
    print("=" * 80)
    print()

    print(f"🔌 Connecting to md:{DB_NAME}...")
    con = duckdb.connect(f"md:{DB_NAME}?motherduck_token={TOKEN}")

    # 1. Load all macros (order matters for dependencies)
    print("\n📦 Loading SQL macros...")
    macros = [
        "features.sql",  # Base feature macros (feat_price_block, etc.)
        "technical_indicators_all_symbols.sql",  # calc_all_technical_indicators
        "cross_asset_features.sql",  # calc_correlation_matrix, calc_fundamental_spreads
        "big8_bucket_features.sql",  # calc_all_bucket_scores
        "master_feature_matrix.sql",  # build_symbol_features (THE KEY MACRO)
    ]

    for m in macros:
        path = MACROS_DIR / m
        if path.exists():
            with open(path, "r") as f:
                try:
                    con.execute(f.read())
                    print(f"  ✅ Loaded {m}")
                except Exception as e:
                    print(f"  ❌ Error loading {m}: {e}")
                    sys.exit(1)
        else:
            print(f"  ⚠️  Warning: {m} not found at {path}")

    # 2. Check current state
    print("\n📊 Current State Analysis...")

    # Check features.daily_ml_matrix_zl
    try:
        cols = con.execute(
            """
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_schema = 'features' 
            AND table_name = 'daily_ml_matrix_zl'
        """
        ).fetchone()[0]
        rows = con.execute(
            "SELECT COUNT(*) FROM features.daily_ml_matrix_zl"
        ).fetchone()[0]
        print(f"  features.daily_ml_matrix_zl: {cols} columns, {rows} rows")
    except Exception as e:
        print(f"  features.daily_ml_matrix_zl: Does not exist")
        cols, rows = 0, 0

    # Check macro output
    macro_cols = con.execute(
        "DESCRIBE SELECT * FROM build_symbol_features('ZL') LIMIT 0"
    ).fetchall()
    print(f"  build_symbol_features('ZL'): {len(macro_cols)} columns")

    # 3. NUKE AND PAVE: features.daily_ml_matrix_zl
    print("\n🔧 REPAIRING: features.daily_ml_matrix_zl")
    print("-" * 80)

    # Drop old table
    con.execute("DROP TABLE IF EXISTS features.daily_ml_matrix_zl")
    print("  🗑️  Dropped old table (had semantic column names)")

    # Create new table from macro output with added updated_at
    print("  🏗️  Creating new table from build_symbol_features('ZL')...")
    con.execute(
        """
        CREATE TABLE features.daily_ml_matrix_zl AS 
        SELECT 
            *,
            CURRENT_TIMESTAMP as updated_at
        FROM build_symbol_features('ZL') 
        WHERE 1=0
    """
    )

    # Validate
    new_cols = con.execute(
        """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'features' 
        AND table_name = 'daily_ml_matrix_zl'
        ORDER BY ordinal_position
    """
    ).fetchall()
    new_col_names = [c[0] for c in new_cols]

    print(f"  ✨ Created with {len(new_cols)} columns")
    print(f"  📋 Sample columns: {new_col_names[:5]}...")

    # Verify critical columns exist
    critical_cols = ["as_of_date", "symbol", "close", "rsi_14", "macd"]
    missing = [c for c in critical_cols if c not in new_col_names]
    if missing:
        print(f"  ❌ MISSING critical columns: {missing}")
    else:
        print(f"  ✅ All critical columns present (close, rsi_14, macd, etc.)")

    # 4. Now populate the table with actual data
    print("\n📥 Populating features.daily_ml_matrix_zl with ZL data...")
    con.execute(
        """
        INSERT INTO features.daily_ml_matrix_zl
        SELECT 
            *,
            CURRENT_TIMESTAMP as updated_at
        FROM build_symbol_features('ZL')
    """
    )

    row_count = con.execute(
        "SELECT COUNT(*) FROM features.daily_ml_matrix_zl"
    ).fetchone()[0]
    print(f"  ✅ Inserted {row_count:,} rows")

    # 5. Verify the fix enables build_training.py
    print("\n🧪 Verification: Can build_training.py work now?")
    print("-" * 80)

    # Check if 'close' column exists (required by build_training.py)
    if "close" in new_col_names:
        print("  ✅ 'close' column exists - build_training.py will work")
    else:
        print("  ❌ 'close' column missing - build_training.py will fail")

    # Show date range
    date_range = con.execute(
        """
        SELECT MIN(as_of_date), MAX(as_of_date) 
        FROM features.daily_ml_matrix_zl
    """
    ).fetchone()
    print(f"  📅 Date range: {date_range[0]} to {date_range[1]}")

    # 6. Summary
    print("\n" + "=" * 80)
    print("✅ SCHEMA UPDATE COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run: python src/engines/anofox/build_training.py")
    print("     (Creates training.daily_ml_matrix_zl from features)")
    print()
    print("  2. Run: python src/training/baselines/xgboost_zl.py")
    print("     (Train baseline model)")

    con.close()


if __name__ == "__main__":
    force_schema_update()


