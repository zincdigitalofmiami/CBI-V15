"""
Database Migration Runner
Applies versioned migrations in order
"""
import duckdb
import os
from pathlib import Path
from datetime import datetime


def get_applied_migrations(conn: duckdb.DuckDBPyConnection) -> set:
    """Get set of already-applied migration versions."""
    try:
        result = conn.execute("""
            SELECT version FROM ops.schema_migrations ORDER BY version
        """).fetchall()
        return {row[0] for row in result}
    except:
        # Table doesn't exist yet
        return set()


def create_migration_table(conn: duckdb.DuckDBPyConnection):
    """Create the migration tracking table."""
    conn.execute("""
        CREATE SCHEMA IF NOT EXISTS ops;
        CREATE TABLE IF NOT EXISTS ops.schema_migrations (
            version VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def apply_migration(conn: duckdb.DuckDBPyConnection, path: Path) -> bool:
    """Apply a single migration file."""
    version = path.stem.split("__")[0]
    name = path.stem.split("__")[1] if "__" in path.stem else path.stem
    
    print(f"  Applying {version}: {name}...")
    
    try:
        sql = path.read_text()
        conn.execute(sql)
        
        # Record migration
        conn.execute("""
            INSERT INTO ops.schema_migrations (version, name)
            VALUES (?, ?)
        """, [version, name])
        
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def run_migrations(conn: duckdb.DuckDBPyConnection, migrations_dir: Path) -> dict:
    """Run all pending migrations."""
    create_migration_table(conn)
    applied = get_applied_migrations(conn)
    
    versions_dir = migrations_dir / "versions"
    if not versions_dir.exists():
        print("No versions directory found")
        return {"applied": 0, "skipped": 0, "failed": 0}
    
    # Get all migration files, sorted by version
    migration_files = sorted(versions_dir.glob("V*.sql"))
    
    results = {"applied": 0, "skipped": 0, "failed": 0}
    
    for path in migration_files:
        version = path.stem.split("__")[0]
        
        if version in applied:
            results["skipped"] += 1
            continue
        
        if apply_migration(conn, path):
            results["applied"] += 1
        else:
            results["failed"] += 1
            break  # Stop on first failure
    
    return results


if __name__ == "__main__":
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        print("Connecting to MotherDuck...")
        conn = duckdb.connect(f"md:usoil_intelligence?motherduck_token={token}")
    else:
        print("Connecting to local DuckDB...")
        conn = duckdb.connect("data/duckdb/cbi_v15.duckdb")
    
    migrations_dir = Path(__file__).parent
    results = run_migrations(conn, migrations_dir)
    
    print("\n=== Migration Results ===")
    print(f"  Applied: {results['applied']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Failed:  {results['failed']}")
    
    conn.close()

