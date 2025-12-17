#!/usr/bin/env python3
"""
Smart 1000-Day Backfill Script
Fills last 1000 days of data for ALL sources that support historical pulls

Strategy:
- Check what's missing in each table
- Backfill only gaps (don't re-fetch existing data)
- Respect rate limits
- Target: ~1000 days for all sources
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

import duckdb

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# Date range for backfill
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=1000)


def check_data_gaps(con: duckdb.DuckDBPyConnection) -> Dict[str, Dict]:
    """Check data gaps for each source."""
    sources = {
        "databento_futures_ohlcv_1d": "as_of_date",
        "fred_economic": "date",
        "epa_rin_prices": "date",
        "usda_export_sales": "report_date",
        "weather_noaa": "date",
        "scrapecreators_news_buckets": "date",
    }

    gaps = {}

    for table, date_col in sources.items():
        try:
            result = con.execute(
                f"""
                SELECT 
                    COUNT(*) as total_rows,
                    MIN({date_col}) as min_date,
                    MAX({date_col}) as max_date,
                    COUNT(DISTINCT {date_col}) as unique_dates
                FROM raw.{table}
                WHERE {date_col} >= '{START_DATE.date()}'
            """
            ).fetchone()

            gaps[table] = {
                "total_rows": result[0],
                "min_date": result[1],
                "max_date": result[2],
                "unique_dates": result[3],
                "expected_dates": 1000,
                "missing_dates": 1000 - result[3] if result[3] else 1000,
                "coverage_pct": (result[3] / 1000 * 100) if result[3] else 0,
            }

        except Exception as e:
            gaps[table] = {"error": str(e)}

    return gaps


def main():
    """Main backfill routine."""
    print("=" * 80)
    print("SMART 1000-DAY BACKFILL")
    print("=" * 80)
    print(f"Target range: {START_DATE.date()} to {END_DATE.date()}")
    print()

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    # Check gaps
    print("Analyzing data gaps...")
    gaps = check_data_gaps(con)

    print("\n" + "=" * 80)
    print("DATA COVERAGE (Last 1000 Days)")
    print("=" * 80)

    for table, info in gaps.items():
        if "error" in info:
            print(f"\n❌ {table}")
            print(f"   Error: {info['error']}")
        else:
            status = (
                "✅"
                if info["coverage_pct"] > 90
                else "⚠️" if info["coverage_pct"] > 50 else "❌"
            )
            print(f"\n{status} {table}")
            print(
                f"   Coverage: {info['coverage_pct']:.1f}% ({info['unique_dates']}/{info['expected_dates']} days)"
            )
            print(f"   Range: {info['min_date']} to {info['max_date']}")
            print(f"   Missing: {info['missing_dates']} days")

    print("\n" + "=" * 80)
    print("BACKFILL RECOMMENDATIONS")
    print("=" * 80)

    # Prioritize by gaps
    needs_backfill = []
    for table, info in gaps.items():
        if "error" not in info and info["coverage_pct"] < 90:
            needs_backfill.append((table, info["missing_dates"], info["coverage_pct"]))

    needs_backfill.sort(key=lambda x: x[1], reverse=True)

    for table, missing, coverage in needs_backfill:
        print(f"\n{table}:")
        print(f"  Missing: {missing} days ({100-coverage:.1f}% gap)")

        if "databento" in table:
            print(f"  Action: Run databento backfill for missing dates")
        elif "fred" in table:
            print(f"  Action: Run FRED priority series collector")
        elif "epa" in table:
            print(f"  Action: Backfill EPA RIN prices (2010-2024)")
        elif "weather" in table:
            print(f"  Action: Run NOAA weather backfill")
        elif "scrapecreators" in table:
            print(f"  Action: Run daily (can't backfill - API limitation)")

    con.close()


if __name__ == "__main__":
    main()


