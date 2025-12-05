#!/usr/bin/env python3
"""
Initialize DuckDB Persistent Secrets - Phase 1.2
Sets up persistent secrets for MotherDuck and S3 (Databento).
"""

import duckdb
import os
from pathlib import Path

DUCKDB_PATH = Path('/Volumes/Satechi Hub/ZL-Intelligence/duckdb/cbi-v15.duckdb')


def setup_secrets(conn: duckdb.DuckDBPyConnection):
    """Setup persistent secrets for MotherDuck and S3."""
    print("Setting up persistent secrets...")
    
    # MotherDuck secret (if token provided)
    motherduck_token = os.getenv('MOTHERDUCK_TOKEN')
    if motherduck_token:
        try:
            conn.execute(f"""
                CREATE PERSISTENT SECRET IF NOT EXISTS motherduck_auth (
                    TYPE MOTHERDUCK,
                    TOKEN '{motherduck_token}'
                )
            """)
            print("  ✓ Created MotherDuck secret")
        except Exception as e:
            print(f"  ⚠ MotherDuck secret error: {e}")
    else:
        print("  ⚠ MOTHERDUCK_TOKEN not set - skipping MotherDuck secret")
    
    # S3 secret (if credentials provided)
    s3_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    s3_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    s3_region = os.getenv('AWS_REGION', 'us-east-1')
    
    if s3_key_id and s3_secret:
        try:
            conn.execute(f"""
                CREATE PERSISTENT SECRET IF NOT EXISTS s3_access (
                    TYPE S3,
                    KEY_ID '{s3_key_id}',
                    SECRET '{s3_secret}',
                    REGION '{s3_region}'
                )
            """)
            print("  ✓ Created S3 secret")
        except Exception as e:
            print(f"  ⚠ S3 secret error: {e}")
    else:
        print("  ⚠ AWS credentials not set - skipping S3 secret")
    
    # List secrets
    try:
        secrets = conn.execute("SELECT name FROM duckdb_secrets()").fetchall()
        if secrets:
            print()
            print("Configured secrets:")
            for secret in secrets:
                print(f"  - {secret[0]}")
    except Exception as e:
        print(f"  Note: Could not list secrets: {e}")


def main():
    """Initialize DuckDB secrets."""
    print("=" * 60)
    print("INITIALIZING DUCKDB SECRETS - Phase 1.2")
    print("=" * 60)
    print(f"DuckDB Path: {DUCKDB_PATH}")
    print()
    
    if not DUCKDB_PATH.exists():
        print(f"ERROR: DuckDB database not found at {DUCKDB_PATH}")
        print("Run create_duckdb_schema.py first!")
        return
    
    conn = duckdb.connect(str(DUCKDB_PATH))
    
    try:
        setup_secrets(conn)
        print()
        print("=" * 60)
        print("SECRETS INITIALIZATION COMPLETE")
        print("=" * 60)
    finally:
        conn.close()


if __name__ == '__main__':
    main()

