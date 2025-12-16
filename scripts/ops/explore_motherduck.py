#!/usr/bin/env python3
"""
MotherDuck Database Explorer
Comprehensive inspection of all schemas, tables, views, and dependencies.

Usage:
    python scripts/ops/explore_motherduck.py              # Full exploration
    python scripts/ops/explore_motherduck.py --schema raw # Specific schema
    python scripts/ops/explore_motherduck.py --deps       # Show dependencies
"""
import os
import sys
from pathlib import Path
import argparse
import duckdb
from typing import Dict, List, Tuple


MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def connect_motherduck():
    """Connect to MotherDuck"""
    if not MOTHERDUCK_TOKEN:
        print("❌ MOTHERDUCK_TOKEN not set")
        sys.exit(1)

    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")


def list_schemas(conn) -> List[str]:
    """List all schemas"""
    query = """
    SELECT schema_name
    FROM information_schema.schemata
    WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_temp')
    ORDER BY schema_name
    """
    return [row[0] for row in conn.execute(query).fetchall()]


def list_tables(conn, schema: str) -> List[Tuple[str, str, int]]:
    """List tables in schema with row counts"""
    query = f"""
    SELECT
        table_name,
        table_type,
        CAST(
            (SELECT COUNT(*) FROM {schema}.{{table_name}})
        AS INTEGER) as row_count
    FROM information_schema.tables
    WHERE table_schema = '{schema}'
    ORDER BY table_name
    """

    tables = []
    for row in conn.execute(f"SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = '{schema}' ORDER BY table_name").fetchall():
        table_name, table_type = row
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {schema}.{table_name}").fetchone()[0]
        except:
            row_count = -1
        tables.append((table_name, table_type, row_count))

    return tables


def get_table_columns(conn, schema: str, table: str) -> List[Tuple[str, str]]:
    """Get columns for a table"""
    query = f"""
    SELECT
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema = '{schema}'
      AND table_name = '{table}'
    ORDER BY ordinal_position
    """
    return conn.execute(query).fetchall()


def analyze_dependencies(conn, schema: str, table: str) -> Dict:
    """Analyze table dependencies (views that reference this table)"""
    # Get view definitions that reference this table
    query = f"""
    SELECT
        table_schema,
        table_name,
        view_definition
    FROM information_schema.views
    WHERE view_definition LIKE '%{schema}.{table}%'
       OR view_definition LIKE '%{table}%'
    """

    refs = conn.execute(query).fetchall()
    return {
        "referenced_by": [(r[0], r[1]) for r in refs]
    }


def explore_full(conn, target_schema: str = None, show_deps: bool = False):
    """Full database exploration"""
    print("=" * 100)
    print(f"MOTHERDUCK DATABASE EXPLORER: {MOTHERDUCK_DB}")
    print("=" * 100)
    print()

    schemas = list_schemas(conn)

    if target_schema:
        if target_schema not in schemas:
            print(f"❌ Schema '{target_schema}' not found")
            return
        schemas = [target_schema]

    print(f"📊 Found {len(schemas)} schema(s)")
    print()

    for schema in schemas:
        print("─" * 100)
        print(f"📁 SCHEMA: {schema}")
        print("─" * 100)

        tables = list_tables(conn, schema)

        if not tables:
            print("  (empty)")
            print()
            continue

        for table_name, table_type, row_count in tables:
            icon = "📋" if table_type == "BASE TABLE" else "👁️"
            count_str = f"{row_count:,}" if row_count >= 0 else "N/A"
            print(f"\n{icon} {table_name} ({table_type}) - {count_str} rows")

            # Show columns
            columns = get_table_columns(conn, schema, table_name)
            print(f"   Columns ({len(columns)}):")
            for col_name, col_type in columns[:10]:  # Limit to first 10
                print(f"     • {col_name}: {col_type}")
            if len(columns) > 10:
                print(f"     ... and {len(columns) - 10} more")

            # Show dependencies if requested
            if show_deps:
                deps = analyze_dependencies(conn, schema, table_name)
                if deps["referenced_by"]:
                    print(f"   Referenced by:")
                    for ref_schema, ref_table in deps["referenced_by"]:
                        print(f"     → {ref_schema}.{ref_table}")

        print()

    print("=" * 100)
    print("✅ EXPLORATION COMPLETE")
    print("=" * 100)


def print_summary(conn):
    """Print database summary"""
    print("=" * 100)
    print(f"DATABASE SUMMARY: {MOTHERDUCK_DB}")
    print("=" * 100)
    print()

    schemas = list_schemas(conn)

    total_tables = 0
    total_views = 0
    total_rows = 0

    for schema in schemas:
        tables = list_tables(conn, schema)
        base_tables = [t for t in tables if t[1] == "BASE TABLE"]
        views = [t for t in tables if t[1] == "VIEW"]
        schema_rows = sum(t[2] for t in tables if t[2] > 0)

        total_tables += len(base_tables)
        total_views += len(views)
        total_rows += schema_rows

        print(f"📁 {schema:20s} | Tables: {len(base_tables):3d} | Views: {len(views):3d} | Rows: {schema_rows:12,d}")

    print()
    print(f"TOTAL: {total_tables} tables, {total_views} views, {total_rows:,} rows across {len(schemas)} schemas")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Explore MotherDuck database")
    parser.add_argument("--schema", help="Specific schema to explore")
    parser.add_argument("--deps", action="store_true", help="Show dependencies")
    parser.add_argument("--summary", action="store_true", help="Show summary only")

    args = parser.parse_args()

    conn = connect_motherduck()

    if args.summary:
        print_summary(conn)
    else:
        explore_full(conn, args.schema, args.deps)

    conn.close()


if __name__ == "__main__":
    main()
