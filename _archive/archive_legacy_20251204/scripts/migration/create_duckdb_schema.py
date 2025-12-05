#!/usr/bin/env python3
"""
Create DuckDB Schema - Phase 1.2
Creates DuckDB database and schema structure matching BigQuery datasets.
"""

import duckdb
import json
from pathlib import Path

DUCKDB_PATH = Path('/Volumes/Satechi Hub/ZL-Intelligence/duckdb/cbi-v15.duckdb')
MANIFEST_FILE = Path('/Volumes/Satechi Hub/CBI-V15/scripts/migration/bq_quick_manifest.json')

# Schema definitions matching BigQuery structure
SCHEMAS = ['raw', 'staging', 'features', 'training', 'forecasts', 'reference', 'ops']


def create_schemas(conn: duckdb.DuckDBPyConnection):
    """Create all schemas."""
    print("Creating schemas...")
    for schema in SCHEMAS:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        print(f"  ✓ Created schema: {schema}")


def create_duckdb_database():
    """Create DuckDB database and schemas."""
    print("=" * 60)
    print("CREATING DUCKDB SCHEMA - Phase 1.2")
    print("=" * 60)
    print(f"DuckDB Path: {DUCKDB_PATH}")
    print()
    
    # Connect to DuckDB (creates file if doesn't exist)
    conn = duckdb.connect(str(DUCKDB_PATH))
    
    try:
        # Create schemas
        create_schemas(conn)
        
        # Verify schemas
        schema_list = conn.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_temp')").fetchall()
        print()
        print("Created schemas:")
        for schema_row in schema_list:
            print(f"  - {schema_row[0]}")
        
        print()
        print("=" * 60)
        print("SCHEMA CREATION COMPLETE")
        print("=" * 60)
        print(f"DuckDB database: {DUCKDB_PATH}")
        print(f"Schemas created: {len(SCHEMAS)}")
        
    finally:
        conn.close()


if __name__ == '__main__':
    create_duckdb_database()

