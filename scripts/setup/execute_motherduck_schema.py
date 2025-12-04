#!/usr/bin/env python3
"""
Execute MotherDuck Schema Creation
Runs all DDL scripts in order to create 8 schemas + ~37 tables
"""
import duckdb
import os
from pathlib import Path

# MotherDuck connection
MOTHERDUCK_TOKEN = os.getenv('MOTHERDUCK_TOKEN')
if not MOTHERDUCK_TOKEN:
    raise ValueError("MOTHERDUCK_TOKEN environment variable not set")

# Schema SQL files directory
SCHEMA_DIR = Path("/Volumes/Satechi Hub/CBI-V15/Data/db/schema")

# Execution order
SCHEMA_FILES = [
    "00_motherduck_init.sql",
    "01_raw_schema.sql",
    "02_raw_staging_schema.sql",
    "03_staging_schema.sql",
    "04_features_schema.sql",
    "05_training_schema.sql",
    "06_forecast_schema.sql",
    "07_reference_schema.sql",
    "08_signals_schema.sql",
    "09_ops_schema.sql",
]

def execute_schema_file(conn, filepath: Path):
    """Execute a SQL schema file"""
    print(f"\n{'='*60}")
    print(f"Executing: {filepath.name}")
    print(f"{'='*60}")
    
    sql = filepath.read_text()
    
    try:
        # Split by semicolon and execute each statement
        statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
        
        for i, stmt in enumerate(statements, 1):
            if stmt:
                print(f"  [{i}/{len(statements)}] {stmt[:80]}...")
                conn.execute(stmt)
        
        print(f"✅ {filepath.name} completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in {filepath.name}: {e}")
        return False

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          MotherDuck Schema Initialization                    ║
║          usoil_intelligence Database                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Connect to MotherDuck
    print("Connecting to MotherDuck...")
    conn = duckdb.connect(f'md:usoil_intelligence?motherduck_token={MOTHERDUCK_TOKEN}')
    print("✅ Connected to MotherDuck\n")
    
    # Execute each schema file
    results = {}
    for filename in SCHEMA_FILES:
        filepath = SCHEMA_DIR / filename
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filename}")
            results[filename] = False
            continue
        
        results[filename] = execute_schema_file(conn, filepath)
    
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    for filename, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status:12} - {filename}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n{passed}/{total} schema files executed successfully")
    
    # Verify schemas created
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    schemas = conn.execute("SELECT schema_name FROM information_schema.schemata ORDER BY schema_name").fetchall()
    print(f"\n✅ Schemas created ({len(schemas)}):")
    for schema in schemas:
        print(f"   - {schema[0]}")
    
    # Count tables per schema
    print(f"\n✅ Tables created:")
    for schema in schemas:
        if schema[0] in ['information_schema', 'pg_catalog', 'main']:
            continue
        table_count = conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema[0]}'").fetchone()[0]
        print(f"   - {schema[0]}: {table_count} tables")
    
    print(f"\n{'='*60}")
    print("✅ MotherDuck schema initialization complete")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

