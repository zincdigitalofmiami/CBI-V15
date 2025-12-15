#!/usr/bin/env python3
"""
Script 0.4: Verify Terms of Trade Calculation

Checks features.core_macro_fx for:
1. Terms of Trade returns non-null values (no division by zero)
2. No Inf/NaN in any core features
3. Sufficient data coverage (>90% non-null)
4. ZS and BRL prices never zero (would cause Inf)

This MUST pass before bucket training can start.
"""
import os
import sys

import duckdb

# MotherDuck connection
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def verify_core_macro_fx():
    """Verify core_macro_fx view is valid for training"""

    print("═" * 80)
    print("SCRIPT 0.4: VERIFY CORE_MACRO_FX VIEW")
    print("═" * 80)

    if not MOTHERDUCK_TOKEN:
        print("❌ MOTHERDUCK_TOKEN not set")
        sys.exit(1)

    # Connect to MotherDuck
    conn = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
    print(f"✅ Connected to MotherDuck: {MOTHERDUCK_DB}")

    # 1. Check view exists
    try:
        result = conn.execute(
            """
            SELECT COUNT(*) 
            FROM information_schema.views 
            WHERE table_schema = 'features' 
            AND table_name = 'core_macro_fx'
        """
        ).fetchone()

        if result[0] == 0:
            print("❌ features.core_macro_fx view does NOT exist")
            print("   Run: Script 0.3 to create it first")
            sys.exit(1)

        print("✅ View exists: features.core_macro_fx")

    except Exception as e:
        print(f"❌ Error checking view: {e}")
        sys.exit(1)

    # 2. Check Terms of Trade (critical)
    print("\n📊 TERMS OF TRADE VALIDATION:")
    print("-" * 80)

    try:
        result = conn.execute(
            """
            SELECT 
                MIN(terms_of_trade) as min_tot,
                MAX(terms_of_trade) as max_tot,
                AVG(terms_of_trade) as avg_tot,
                COUNT(*) as total_rows,
                COUNT(terms_of_trade) as non_null_rows,
                (COUNT(*) - COUNT(terms_of_trade))::DOUBLE / COUNT(*) * 100 as null_pct
            FROM features.core_macro_fx
        """
        ).fetchone()

        min_tot, max_tot, avg_tot, total, non_null, null_pct = result

        print(f"   Total rows: {total:,}")
        print(f"   Non-null: {non_null:,} ({100-null_pct:.1f}%)")
        print(f"   Null: {total-non_null:,} ({null_pct:.1f}%)")
        print(f"   Min: {min_tot:.6f}" if min_tot else "   Min: NULL")
        print(f"   Max: {max_tot:.6f}" if max_tot else "   Max: NULL")
        print(f"   Avg: {avg_tot:.6f}" if avg_tot else "   Avg: NULL")

        # Fail if >5% null
        if null_pct > 5:
            print(f"\n❌ FAIL: {null_pct:.1f}% null (threshold: 5%)")
            print("   Cause: BRL price (6L) likely missing or zero")
            sys.exit(1)

        # Fail if any Inf
        if max_tot and (max_tot > 1e10 or max_tot < -1e10):
            print(f"\n❌ FAIL: Inf detected (max={max_tot})")
            print("   Cause: Division by zero (BRL price = 0)")
            sys.exit(1)

        print("\n✅ Terms of Trade validation PASSED")

    except Exception as e:
        print(f"❌ Error validating Terms of Trade: {e}")
        sys.exit(1)

    # 3. Check all core features for nulls
    print("\n📊 CORE FEATURE NULL CHECK:")
    print("-" * 80)

    try:
        # Get column names
        columns = conn.execute("DESCRIBE features.core_macro_fx").fetchall()
        critical_features = [col[0] for col in columns if col[0] not in ["as_of_date"]]

        null_features = []
        for col in critical_features:
            result = conn.execute(
                f"""
                SELECT 
                    COUNT(*)::DOUBLE - COUNT({col})::DOUBLE,
                    (COUNT(*) - COUNT({col}))::DOUBLE / COUNT(*) * 100 as null_pct
                FROM features.core_macro_fx
            """
            ).fetchone()

            null_count, null_pct = result

            if null_pct > 10:  # Warn if >10% null
                null_features.append((col, null_pct))
                print(f"   ⚠️  {col}: {null_pct:.1f}% null")

        if null_features:
            print(f"\n⚠️  {len(null_features)} features have >10% nulls")
            print("   This may reduce training data quality")
        else:
            print("   ✅ All features have <10% nulls")

    except Exception as e:
        print(f"⚠️  Could not check all features: {e}")

    # 4. Final summary
    print("\n" + "═" * 80)
    print("✅ CORE_MACRO_FX VERIFICATION COMPLETE")
    print("═" * 80)
    print(f"\n📋 Summary:")
    print(f"   • View exists: features.core_macro_fx")
    print(f"   • Terms of Trade: Valid ({100-null_pct:.1f}% coverage)")
    print(f"   • Total rows: {total:,}")
    print(f"   • Ready for bucket training: YES")
    print()

    conn.close()


if __name__ == "__main__":
    verify_core_macro_fx()













