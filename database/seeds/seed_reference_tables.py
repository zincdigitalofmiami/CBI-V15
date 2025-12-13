"""
Seed: All Reference Tables
Master script to seed all reference data
"""
import duckdb
import os

from seed_symbols import seed_symbols
from seed_regimes import seed_regime_weights
from seed_splits import seed_splits


def seed_all(conn: duckdb.DuckDBPyConnection) -> dict:
    """Seed all reference tables."""
    results = {}
    
    # Run DDL for reference schema first
    print("Creating reference tables...")
    
    # Seed in order
    print("Seeding symbols...")
    results["symbols"] = seed_symbols(conn)
    
    print("Seeding regime weights...")
    results["regime_weights"] = seed_regime_weights(conn)
    
    print("Seeding splits...")
    results["splits"] = seed_splits(conn)
    
    return results


if __name__ == "__main__":
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        print("Connecting to MotherDuck...")
        conn = duckdb.connect(f"md:usoil_intelligence?motherduck_token={token}")
    else:
        print("Connecting to local DuckDB...")
        conn = duckdb.connect("data/duckdb/cbi_v15.duckdb")
    
    results = seed_all(conn)
    
    print("\n=== Seed Results ===")
    for table, count in results.items():
        print(f"  {table}: {count} rows")
    
    conn.close()
    print("\nDone!")

