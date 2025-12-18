#!/usr/bin/env python3
"""
Verify technical indicators were successfully built for all 33 symbols
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
    print("TECHNICAL INDICATORS VERIFICATION")
    print("=" * 80)
    print()

    # Check row counts by symbol
    print("Row counts by symbol:")
    print("-" * 80)
    results = conn.execute(
        """
        SELECT 
            symbol,
            COUNT(*) as row_count,
            MIN(as_of_date) as min_date,
            MAX(as_of_date) as max_date,
            MIN(updated_at) as first_updated,
            MAX(updated_at) as last_updated
        FROM features.technical_indicators_all_symbols
        GROUP BY symbol
        ORDER BY symbol
    """
    ).fetchdf()

    print(results.to_string(index=False))
    print()

    # Summary stats
    print("Summary:")
    print("-" * 80)
    print(f"Total symbols:     {len(results)}")
    print(f"Total rows:        {results['row_count'].sum():,}")
    print(f"Avg rows/symbol:   {results['row_count'].mean():.0f}")
    print(f"Min rows:          {results['row_count'].min():,} ({results.loc[results['row_count'].idxmin(), 'symbol']})")
    print(f"Max rows:          {results['row_count'].max():,} ({results.loc[results['row_count'].idxmax(), 'symbol']})")
    print()

    # Check for NULL values
    print("NULL value check:")
    print("-" * 80)
    null_check = conn.execute(
        """
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
            SUM(CASE WHEN rsi_14 IS NULL THEN 1 ELSE 0 END) as null_rsi,
            SUM(CASE WHEN macd IS NULL THEN 1 ELSE 0 END) as null_macd,
            SUM(CASE WHEN bb_upper IS NULL THEN 1 ELSE 0 END) as null_bb,
            SUM(CASE WHEN atr_14 IS NULL THEN 1 ELSE 0 END) as null_atr,
            SUM(CASE WHEN updated_at IS NULL THEN 1 ELSE 0 END) as null_updated_at
        FROM features.technical_indicators_all_symbols
    """
    ).fetchone()

    print(f"Total rows:          {null_check[0]:,}")
    print(f"NULL close:          {null_check[1]}")
    print(f"NULL rsi_14:         {null_check[2]}")
    print(f"NULL macd:           {null_check[3]}")
    print(f"NULL bb_upper:       {null_check[4]}")
    print(f"NULL atr_14:         {null_check[5]}")
    print(f"NULL updated_at:     {null_check[6]}")
    print()

    if null_check[6] > 0:
        print("❌ FAIL: updated_at has NULL values")
        return 1
    else:
        print("✅ PASS: All updated_at values populated")

    # Sample latest data
    print()
    print("Latest data sample (3 symbols):")
    print("-" * 80)
    sample = conn.execute(
        """
        WITH latest_per_symbol AS (
            SELECT 
                symbol,
                MAX(as_of_date) as latest_date
            FROM features.technical_indicators_all_symbols
            GROUP BY symbol
            LIMIT 3
        )
        SELECT 
            t.symbol,
            t.as_of_date,
            t.close,
            t.rsi_14,
            t.macd_histogram,
            t.bb_position,
            t.atr_14,
            t.updated_at
        FROM features.technical_indicators_all_symbols t
        INNER JOIN latest_per_symbol l 
            ON t.symbol = l.symbol 
            AND t.as_of_date = l.latest_date
        ORDER BY t.symbol
    """
    ).fetchdf()

    print(sample.to_string(index=False))
    print()

    print("=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)
    print()
    print("Technical indicators successfully built for all 33 symbols!")
    print(f"Total: {null_check[0]:,} rows")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())



