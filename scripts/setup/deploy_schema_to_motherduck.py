import duckdb
import os
from pathlib import Path

# Configuration
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB_NAME = "cbi_v15"


def run():
    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN environment variable not set")

    print(f"Connecting to MotherDuck: md:{DB_NAME}...")
    # Connect to MotherDuck
    con = duckdb.connect(f"md:{DB_NAME}?motherduck_token={MOTHERDUCK_TOKEN}")

    # Schema Directory
    # We use the same definitions as local DuckDB
    ROOT_DIR = Path(__file__).resolve().parents[2]
    SCHEMA_DIR = ROOT_DIR / "database" / "definitions"

    if not SCHEMA_DIR.exists():
        raise FileNotFoundError(f"Schema directory not found: {SCHEMA_DIR}")

    # 1. Init Schemas
    print("Initializing Schemas...")
    init_script = SCHEMA_DIR / "00_init" / "00_schemas.sql"
    if init_script.exists():
        print(f"  Executing {init_script.name}...")
        con.execute(init_script.read_text())

    # 2. Create Tables (Iterate through layers)
    layers = [
        "01_raw",
        "02_staging",
        "03_features",
        "04_training",
        "05_forecasts",
        "06_reference",
        "07_ops",
        "08_tsci",
    ]

    for layer in layers:
        layer_dir = SCHEMA_DIR / layer
        if layer_dir.exists():
            print(f"Processing layer: {layer}...")
            for sql_file in sorted(layer_dir.glob("*.sql")):
                print(f"  Executing {sql_file.name}...")
                try:
                    # Read and execute DDL
                    # Note: We might need to handle 'CREATE OR REPLACE' if not present,
                    # but our files should be idempotent or we rely on MD to handle it.
                    # Ideally, we use CREATE TABLE IF NOT EXISTS or CREATE OR REPLACE.
                    sql = sql_file.read_text()
                    con.execute(sql)
                except Exception as e:
                    print(f"  ❌ Error executing {sql_file.name}: {e}")

    print("✅ MotherDuck Schema Deployment Complete.")
    con.close()


if __name__ == "__main__":
    run()
