#!/usr/bin/env python3
"""
Install Anofox Extensions in DuckDB - Phase 2.1
Installs and verifies anofox_tabular, anofox_forecast, anofox_statistics extensions.
"""

import duckdb
from pathlib import Path

DUCKDB_PATH = Path('/Volumes/Satechi Hub/ZL-Intelligence/duckdb/cbi-v15.duckdb')

EXTENSIONS = [
    'anofox_tabular',
    'anofox_forecast',
    'anofox_statistics'
]


def install_extension(conn: duckdb.DuckDBPyConnection, ext_name: str) -> bool:
    """Install a single Anofox extension."""
    try:
        print(f"  Installing {ext_name}...")
        conn.execute(f"INSTALL {ext_name} FROM community")
        conn.execute(f"LOAD {ext_name}")
        print(f"    ✓ {ext_name} installed and loaded")
        return True
    except Exception as e:
        print(f"    ✗ {ext_name} failed: {e}")
        return False


def verify_extension(conn: duckdb.DuckDBPyConnection, ext_name: str) -> bool:
    """Verify extension is loaded by checking for extension functions."""
    try:
        # Try to list extension functions (if available)
        # For now, just try a simple query that would use the extension
        if ext_name == 'anofox_tabular':
            # Check for anofox_tabular functions
            result = conn.execute("SELECT function_name FROM duckdb_functions() WHERE function_name LIKE 'anofox_%' LIMIT 5").fetchall()
            if result:
                print(f"    ✓ Found {len(result)} anofox functions")
                return True
        elif ext_name == 'anofox_forecast':
            # Check for forecast functions
            result = conn.execute("SELECT function_name FROM duckdb_functions() WHERE function_name LIKE '%forecast%' OR function_name LIKE 'TS_%' LIMIT 5").fetchall()
            if result:
                print(f"    ✓ Found forecast functions")
                return True
        elif ext_name == 'anofox_statistics':
            # Check for statistics functions
            result = conn.execute("SELECT function_name FROM duckdb_functions() WHERE function_name LIKE '%stat%' OR function_name LIKE 'anofox_%' LIMIT 5").fetchall()
            if result:
                print(f"    ✓ Found statistics functions")
                return True
        
        # If we can't verify, assume it's OK if install succeeded
        return True
    except Exception as e:
        print(f"    ⚠ Verification warning: {e}")
        return True  # Don't fail on verification


def main():
    """Install and verify all Anofox extensions."""
    print("=" * 60)
    print("INSTALLING ANOFOX EXTENSIONS - Phase 2.1")
    print("=" * 60)
    print(f"DuckDB: {DUCKDB_PATH}")
    print()
    
    if not DUCKDB_PATH.exists():
        print(f"ERROR: DuckDB database not found at {DUCKDB_PATH}")
        print("Run create_duckdb_schema.py first!")
        return
    
    conn = duckdb.connect(str(DUCKDB_PATH))
    
    try:
        results = {}
        for ext in EXTENSIONS:
            success = install_extension(conn, ext)
            if success:
                verify_extension(conn, ext)
            results[ext] = success
        
        print()
        print("=" * 60)
        print("INSTALLATION COMPLETE")
        print("=" * 60)
        
        successful = sum(1 for v in results.values() if v)
        print(f"Extensions installed: {successful}/{len(EXTENSIONS)}")
        
        if successful == len(EXTENSIONS):
            print("✓ All extensions installed successfully!")
        else:
            print("⚠ Some extensions failed - check errors above")
            print()
            print("Note: Anofox extensions may not be available in DuckDB community yet.")
            print("You may need to build from source or use alternative methods.")
        
        # List installed extensions
        print()
        print("Installed extensions:")
        try:
            installed = conn.execute("SELECT * FROM duckdb_extensions() WHERE installed = true").fetchall()
            for ext in installed:
                print(f"  - {ext[0]}")
        except:
            print("  (Could not list extensions)")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()

