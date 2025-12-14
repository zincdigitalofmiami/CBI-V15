#!/usr/bin/env python3
"""
SQL Smoke Tests Runner (MotherDuck or local DuckDB)

Runs a minimal set of invariants to validate that key tables exist and have
recent/fresh data. Uses the existing database/tests/harness.py for SQL files
under database/tests/sql/ if present; otherwise runs a built-in quick check.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "database" / "tests"


def connect(motherduck: bool) -> duckdb.DuckDBPyConnection:
    db = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    token = os.getenv("MOTHERDUCK_TOKEN")
    if motherduck and token:
        return duckdb.connect(f"md:{db}?motherduck_token={token}")
    # fallback to local
    db_path = ROOT / "data" / "duckdb" / "cbi_v15.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def built_in_quick_checks(conn: duckdb.DuckDBPyConnection) -> bool:
    """Run a couple of quick invariants without external SQL files."""
    ok = True
    try:
        # Check that forecasts table exists and has any rows
        conn.execute(
            """
            CREATE TEMP TABLE _exists AS
            SELECT COUNT(*) AS cnt
            FROM information_schema.tables
            WHERE table_schema = 'forecasts' AND table_name = 'zl_predictions';
            """
        )
        cnt = conn.execute("SELECT cnt FROM _exists").fetchone()[0]
        if cnt == 0:
            print("[FAIL] forecasts.zl_predictions table missing")
            ok = False
        else:
            print("[PASS] forecasts.zl_predictions exists")

        # If exists, check freshness (<= 14 days old)
        if cnt:
            row = conn.execute(
                "SELECT max(as_of_date) FROM forecasts.zl_predictions"
            ).fetchone()
            max_date = row[0]
            if max_date is None:
                print("[FAIL] forecasts.zl_predictions has no data")
                ok = False
            else:
                try:
                    # duckdb returns Python date for DATE type
                    delta = datetime.utcnow().date() - max_date
                    if delta.days <= 14:
                        print(
                            f"[PASS] forecasts.zl_predictions fresh (latest {max_date})"
                        )
                    else:
                        print(
                            f"[WARN] forecasts.zl_predictions stale (latest {max_date})"
                        )
                except Exception:
                    print("[INFO] Unable to compute freshness; non-date type")
    except Exception as e:
        print(f"[ERROR] Built-in checks failed: {e}")
        ok = False
    return ok


def main():
    motherduck = "--motherduck" in sys.argv
    conn = connect(motherduck)

    sql_dir = TESTS_DIR / "sql"
    if sql_dir.exists() and any(sql_dir.glob("test_*.sql")):
        # Use harness if tests exist
        sys.path.insert(0, str(TESTS_DIR))
        from harness import run_all_tests, print_summary  # type: ignore

        results = run_all_tests(conn, TESTS_DIR)
        success = print_summary(results)
        conn.close()
        sys.exit(0 if success else 1)
    else:
        # Run built-in minimal checks
        success = built_in_quick_checks(conn)
        conn.close()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
