#!/usr/bin/env python3
"""
Execute Local DuckDB Schema Creation
Runs all DDL scripts in order from the definitions directory.
This script ONLY creates the schema structure (tables, views).
It does NOT populate features or training data.
"""
import duckdb
from pathlib import Path
import glob

# Local DuckDB connection
DB_DIR = Path("data/duckdb")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "cbi_v15.duckdb"

# Schema SQL files directory
SCHEMA_DIR = Path("database/definitions")


def execute_statements(conn, statements, statement_type):
    """Execute a list of SQL statements of a given type."""
    print(f"\n{'='*60}")
    print(f"Executing {len(statements)} {statement_type} statements...")
    print(f"{'='*60}")

    success_count = 0
    for i, stmt in enumerate(statements, 1):
        # Preview the first line of the statement
        preview = stmt.strip().split("\n")[0][:80]
        try:
            print(f"  [{i}/{len(statements)}] {preview}...")
            conn.execute(stmt)
            success_count += 1
        except Exception as e:
            print(f"  ❌ FAILED: {preview}")
            print(f"     Error: {e}")

    print(
        f"✅ Executed {success_count}/{len(statements)} {statement_type} statements successfully."
    )
    return success_count == len(statements)


def main():
    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║          Local DuckDB Schema Initialization                  ║
║          Database: {DB_PATH}                                 ║
╚══════════════════════════════════════════════════════════════╝
    """
    )

    # Connect to Local DuckDB
    print(f"Connecting to local DuckDB at {DB_PATH}...")
    conn = duckdb.connect(str(DB_PATH), config={"allow_unsigned_extensions": "true"})
    print("✅ Connected to Local DuckDB")

    # Find all SQL files and sort them to ensure `00_init` runs first
    sql_files = sorted(glob.glob(f"{str(SCHEMA_DIR)}/**/*.sql", recursive=True))

    create_schema_statements = []
    create_table_statements = []
    other_statements = []

    # Separate statements into three categories for ordered execution
    for filepath_str in sql_files:
        filepath = Path(filepath_str)
        sql = filepath.read_text()

        # Split file content into individual statements
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        for stmt in statements:
            # Clean statement by removing comments for accurate type detection
            clean_stmt = "\n".join(
                [line for line in stmt.split("\n") if not line.strip().startswith("--")]
            ).strip()

            if not clean_stmt:
                continue

            # Categorize the original statement (with comments)
            if clean_stmt.upper().startswith("CREATE SCHEMA"):
                create_schema_statements.append(stmt)
            elif clean_stmt.upper().startswith(
                "CREATE TABLE"
            ) or clean_stmt.upper().startswith("CREATE OR REPLACE TABLE"):
                create_table_statements.append(stmt)
            else:
                other_statements.append(stmt)

    # Execute statements in the correct order
    execute_statements(conn, create_schema_statements, "CREATE SCHEMA")
    execute_statements(conn, create_table_statements, "CREATE TABLE")
    execute_statements(conn, other_statements, "other")

    # --- VERIFICATION ---
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    # List schemas
    schemas = conn.execute(
        "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name"
    ).fetchall()
    print("✅ Schemas created:")
    for schema in schemas:
        print(f"   - {schema[0]}")

    print(f"\n✅ Tables created:")
    for schema_tuple in schemas:
        schema_name = schema_tuple[0]
        if schema_name not in ["information_schema", "pg_catalog", "main"]:
            table_count_result = conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema_name}'"
            ).fetchone()
            if table_count_result:
                print(f"   - {schema_name}: {table_count_result[0]} tables")

    print(f"\n{'='*60}")
    print("✅ Local DuckDB schema initialization complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
