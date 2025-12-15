#!/usr/bin/env python3
"""
Endpoint Verification (Headless)

Runs the same SQL as dashboard/app/api/live/zl/route.ts against MotherDuck.
Requires MOTHERDUCK_TOKEN. Exits 0 with summary if successful; non-zero on error.
"""

import os
import sys
import duckdb


def main() -> None:
    token = os.getenv("MOTHERDUCK_TOKEN")
    dbname = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    if not token:
        print("[endpoint-verify] Skipped: MOTHERDUCK_TOKEN not set")
        sys.exit(0)

    con = duckdb.connect(f"md:{dbname}?motherduck_token={token}")
    sql = (
        "SELECT COUNT(*) AS cnt, MIN(as_of_date) AS min_date, MAX(as_of_date) AS max_date "
        "FROM raw.databento_futures_ohlcv_1d WHERE symbol = 'ZL'"
    )
    try:
        row = con.execute(sql).fetchone()
        cnt, min_date, max_date = row
        print(f"[endpoint-verify] ZL rows={cnt}, min_date={min_date}, max_date={max_date}")
        # Hard fail if table/query resolves but zero rows
        if cnt == 0:
            print("[endpoint-verify] ERROR: Query succeeded but returned zero rows")
            sys.exit(2)
        sys.exit(0)
    except Exception as e:
        print(f"[endpoint-verify] ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
