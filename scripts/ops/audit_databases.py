#!/usr/bin/env python3
"""
Comprehensive Database Audit
Audits both MotherDuck cloud database and local DuckDB files.

Usage:
    python scripts/ops/audit_databases.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import duckdb

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR / ".env.local", override=True)
except ImportError:
    pass

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# Expected schemas
EXPECTED_SCHEMAS = [
    "raw", "staging", "features", "features_dev",
    "training", "forecasts", "reference", "ops", "explanations"
]

# Expected table counts by schema (from DDL analysis)
EXPECTED_TABLE_COUNTS = {
    "reference": 11,
    "raw": 13,
    "staging": 7,
    "features": 6,
    "training": 6,
    "forecasts": 4,
    "ops": 7,
    "explanations": 1
}


def get_motherduck_connection():
    """Get MotherDuck connection"""
    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN environment variable not set")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")


def get_local_connection(db_path: Path):
    """Get local DuckDB connection"""
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path))


def audit_database(conn, db_name: str):
    """Audit a database connection"""
    print(f"\n{'=' * 80}")
    print(f"AUDIT: {db_name}")
    print(f"{'=' * 80}")

    audit_results = {
        "name": db_name,
        "schemas": {},
        "tables": {},
        "row_counts": {},
        "issues": []
    }

    # 1. Check schemas
    print("\n📦 SCHEMAS:")
    try:
        schemas_result = conn.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT LIKE 'pg_%'
            AND schema_name NOT IN ('information_schema', 'main')
            ORDER BY schema_name
        """).fetchall()

        found_schemas = [s[0] for s in schemas_result]
        audit_results["schemas"] = found_schemas

        for schema in EXPECTED_SCHEMAS:
            if schema in found_schemas:
                print(f"  ✅ {schema}")
            else:
                print(f"  ❌ {schema} (MISSING)")
                audit_results["issues"].append(f"Schema '{schema}' is missing")

        # Report unexpected schemas
        unexpected = set(found_schemas) - set(EXPECTED_SCHEMAS)
        for schema in unexpected:
            print(f"  ⚠️  {schema} (UNEXPECTED)")

    except Exception as e:
        print(f"  ❌ Error checking schemas: {e}")
        audit_results["issues"].append(f"Schema check failed: {e}")
        return audit_results

    # 2. Check tables in each schema
    print("\n📊 TABLES BY SCHEMA:")
    total_tables = 0

    for schema in found_schemas:
        if schema not in EXPECTED_SCHEMAS:
            continue

        try:
            tables_result = conn.execute(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = '{schema}'
                ORDER BY table_name
            """).fetchall()

            table_names = [t[0] for t in tables_result]
            audit_results["tables"][schema] = table_names

            expected_count = EXPECTED_TABLE_COUNTS.get(schema, "?")
            actual_count = len(table_names)
            total_tables += actual_count

            status = "✅" if (isinstance(expected_count, int) and actual_count == expected_count) else "⚠️"
            print(f"\n  {status} {schema.upper()}: {actual_count} tables (expected: {expected_count})")

            # Get row counts for each table
            for table_name in table_names:
                try:
                    row_count_result = conn.execute(f"""
                        SELECT COUNT(*) FROM {schema}.{table_name}
                    """).fetchone()
                    row_count = row_count_result[0] if row_count_result else 0

                    audit_results["row_counts"][f"{schema}.{table_name}"] = row_count

                    # Show row count with emoji indicator
                    if row_count == 0:
                        print(f"    📭 {table_name}: 0 rows (EMPTY)")
                    elif row_count < 100:
                        print(f"    ⚠️  {table_name}: {row_count:,} rows (LOW)")
                    else:
                        print(f"    ✅ {table_name}: {row_count:,} rows")

                except Exception as e:
                    print(f"    ❌ {table_name}: Error counting rows - {e}")
                    audit_results["row_counts"][f"{schema}.{table_name}"] = -1

            if actual_count != expected_count and isinstance(expected_count, int):
                audit_results["issues"].append(
                    f"Schema '{schema}' has {actual_count} tables, expected {expected_count}"
                )

        except Exception as e:
            print(f"  ❌ Error checking tables in {schema}: {e}")
            audit_results["issues"].append(f"Table check failed for {schema}: {e}")

    print(f"\n  📊 TOTAL: {total_tables} tables found")

    # 3. Check key feature tables
    print("\n🎯 KEY FEATURE TABLES:")
    key_tables = [
        ("features", "daily_ml_matrix_zl"),
        ("features", "bucket_scores"),
        ("forecasts", "zl_predictions"),
        ("staging", "ohlcv_daily")
    ]

    for schema, table in key_tables:
        if schema not in found_schemas:
            print(f"  ❌ {schema}.{table} (schema missing)")
            continue

        try:
            # Check if table exists
            exists = conn.execute(f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = '{schema}' AND table_name = '{table}'
            """).fetchone()[0]

            if exists:
                # Get details
                details = conn.execute(f"""
                    SELECT
                        COUNT(*) as row_count,
                        (SELECT COUNT(*)
                         FROM information_schema.columns
                         WHERE table_schema = '{schema}' AND table_name = '{table}') as col_count
                    FROM {schema}.{table}
                """).fetchone()

                row_count, col_count = details
                status = "✅" if row_count > 0 else "📭"
                print(f"  {status} {schema}.{table}: {row_count:,} rows, {col_count} columns")

                # Try to get date range if as_of_date exists
                try:
                    date_range = conn.execute(f"""
                        SELECT MIN(as_of_date), MAX(as_of_date)
                        FROM {schema}.{table}
                    """).fetchone()
                    if date_range and date_range[0]:
                        print(f"     📅 Date range: {date_range[0]} to {date_range[1]}")
                except:
                    pass  # Column might not exist

            else:
                print(f"  ❌ {schema}.{table} (NOT FOUND)")
                audit_results["issues"].append(f"Key table {schema}.{table} is missing")

        except Exception as e:
            print(f"  ❌ {schema}.{table}: {e}")

    print(f"\n{'=' * 80}")

    return audit_results


def compare_databases(md_results, local_results):
    """Compare MotherDuck and local database results"""
    print(f"\n{'=' * 80}")
    print("COMPARISON: MotherDuck vs Local")
    print(f"{'=' * 80}")

    print("\n📊 SCHEMA COMPARISON:")
    md_schemas = set(md_results.get("schemas", []))
    local_schemas = set(local_results.get("schemas", []))

    missing_in_local = md_schemas - local_schemas
    missing_in_md = local_schemas - md_schemas
    common = md_schemas & local_schemas

    print(f"  ✅ Common schemas: {len(common)}")
    if missing_in_local:
        print(f"  ⚠️  In MotherDuck only: {', '.join(sorted(missing_in_local))}")
    if missing_in_md:
        print(f"  ⚠️  In Local only: {', '.join(sorted(missing_in_md))}")

    print("\n📊 TABLE COMPARISON:")
    for schema in sorted(common):
        md_tables = set(md_results.get("tables", {}).get(schema, []))
        local_tables = set(local_results.get("tables", {}).get(schema, []))

        if md_tables == local_tables:
            print(f"  ✅ {schema}: {len(md_tables)} tables (SYNCED)")
        else:
            missing_in_local = md_tables - local_tables
            missing_in_md = local_tables - md_tables
            print(f"  ⚠️  {schema}: MD={len(md_tables)}, Local={len(local_tables)}")
            if missing_in_local:
                print(f"     Missing in local: {', '.join(sorted(missing_in_local))}")
            if missing_in_md:
                print(f"     Missing in MD: {', '.join(sorted(missing_in_md))}")

    print("\n📊 ROW COUNT COMPARISON (sample):")
    md_rows = md_results.get("row_counts", {})
    local_rows = local_results.get("row_counts", {})

    # Sample key tables
    sample_tables = [
        "raw.databento_futures_ohlcv_1d",
        "staging.ohlcv_daily",
        "features.daily_ml_matrix_zl",
        "reference.symbols"
    ]

    for table in sample_tables:
        md_count = md_rows.get(table, "N/A")
        local_count = local_rows.get(table, "N/A")

        if md_count == "N/A" and local_count == "N/A":
            print(f"  ❌ {table}: Not found in either database")
        elif md_count == local_count:
            print(f"  ✅ {table}: {md_count:,} rows (SYNCED)")
        else:
            md_str = f"{md_count:,}" if isinstance(md_count, int) else str(md_count)
            local_str = f"{local_count:,}" if isinstance(local_count, int) else str(local_count)
            print(f"  ⚠️  {table}: MD={md_str}, Local={local_str}")

    print(f"\n{'=' * 80}")


def main():
    """Main audit function"""
    print("=" * 80)
    print("CBI-V15 DATABASE AUDIT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Audit MotherDuck
    print("\n🌍 Auditing MotherDuck cloud database...")
    try:
        md_conn = get_motherduck_connection()
        md_results = audit_database(md_conn, f"MotherDuck ({MOTHERDUCK_DB})")
        md_conn.close()
    except Exception as e:
        print(f"❌ Failed to audit MotherDuck: {e}")
        md_results = {"issues": [f"Failed to connect: {e}"]}

    # Audit primary local database
    print("\n💾 Auditing primary local database...")
    local_db_path = ROOT_DIR / "data" / "duckdb" / "cbi_v15.duckdb"
    try:
        local_conn = get_local_connection(local_db_path)
        if local_conn:
            local_results = audit_database(local_conn, f"Local DuckDB ({local_db_path.name})")
            local_conn.close()
        else:
            print(f"❌ Local database not found: {local_db_path}")
            local_results = {"issues": ["Database file not found"]}
    except Exception as e:
        print(f"❌ Failed to audit local database: {e}")
        local_results = {"issues": [f"Failed to connect: {e}"]}

    # Compare databases
    if md_results and local_results:
        compare_databases(md_results, local_results)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    print("\n📋 MotherDuck Issues:")
    if md_results.get("issues"):
        for issue in md_results["issues"]:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ No issues found")

    print("\n📋 Local Database Issues:")
    if local_results.get("issues"):
        for issue in local_results["issues"]:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ No issues found")

    # Check other local databases
    print("\n💾 OTHER LOCAL DUCKDB FILES:")
    other_dbs = [
        ROOT_DIR / "src" / "data" / "duckdb" / "cbi_v15.duckdb",
        ROOT_DIR / "archive" / "Data" / "duckdb" / "cbi_v15.duckdb"
    ]

    for db_path in other_dbs:
        if db_path.exists():
            stat = db_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  📁 {db_path.relative_to(ROOT_DIR)}")
            print(f"     Size: {size_mb:.2f} MB, Modified: {mod_time}")
        else:
            print(f"  ❌ {db_path.relative_to(ROOT_DIR)} (not found)")

    print("\n📌 RECOMMENDATIONS:")
    print("  1. Run sync if MotherDuck and Local have different row counts:")
    print("     python scripts/sync_motherduck_to_local.py")
    print("  2. Consolidate local DB files to use only data/duckdb/cbi_v15.duckdb")
    print("  3. Remove or archive old local DB files (src/ and archive/)")
    print("  4. Ensure all ingestion jobs write to MotherDuck (md:cbi_v15)")

    print(f"\n{'=' * 80}")
    print(f"Audit completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
