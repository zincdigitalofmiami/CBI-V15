"""
Test Harness
Runs all SQL tests and reports results
"""
import duckdb
import os
from pathlib import Path
from datetime import datetime


def run_sql_test(conn: duckdb.DuckDBPyConnection, test_file: Path) -> dict:
    """Run a single SQL test file."""
    test_name = test_file.stem
    
    try:
        sql = test_file.read_text()
        result = conn.execute(sql).fetchdf()
        
        # Check for test_result column
        if 'test_result' in result.columns:
            test_result = result['test_result'].iloc[0]
            return {
                "name": test_name,
                "status": test_result,
                "details": result.to_dict('records')[0] if len(result) == 1 else result.to_dict('records')
            }
        else:
            # Return raw results
            return {
                "name": test_name,
                "status": "INFO",
                "details": result.to_dict('records')
            }
    except Exception as e:
        return {
            "name": test_name,
            "status": "ERROR",
            "details": str(e)
        }


def run_all_tests(conn: duckdb.DuckDBPyConnection, tests_dir: Path) -> list:
    """Run all SQL tests in the directory."""
    sql_dir = tests_dir / "sql"
    if not sql_dir.exists():
        print("No sql directory found")
        return []
    
    test_files = sorted(sql_dir.glob("test_*.sql"))
    results = []
    
    for test_file in test_files:
        print(f"Running {test_file.stem}...", end=" ")
        result = run_sql_test(conn, test_file)
        results.append(result)
        print(result["status"])
    
    return results


def print_summary(results: list):
    """Print test summary."""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    warnings = sum(1 for r in results if r["status"] == "WARN")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"  PASSED:   {passed}")
    print(f"  FAILED:   {failed}")
    print(f"  WARNINGS: {warnings}")
    print(f"  SKIPPED:  {skipped}")
    print(f"  ERRORS:   {errors}")
    print("=" * 50)
    
    if failed > 0 or errors > 0:
        print("\nFailed/Error tests:")
        for r in results:
            if r["status"] in ("FAIL", "ERROR"):
                print(f"  - {r['name']}: {r['details']}")
    
    return failed == 0 and errors == 0


if __name__ == "__main__":
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        print("Connecting to MotherDuck...")
        conn = duckdb.connect(f"md:usoil_intelligence?motherduck_token={token}")
    else:
        print("Connecting to local DuckDB...")
        conn = duckdb.connect("data/duckdb/cbi_v15.duckdb")
    
    tests_dir = Path(__file__).parent
    results = run_all_tests(conn, tests_dir)
    
    success = print_summary(results)
    conn.close()
    
    exit(0 if success else 1)

