import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from cbi_utils.bigquery_client import get_client


def inspect_table(client, table_id):
    try:
        table = client.get_table(table_id)
        print(f"\nTABLE: {table_id}")
        print(f"  Rows: {table.num_rows}")
        print(
            f"  Partitioning: {table.partitioning_type if table.partitioning_type else 'None'}"
        )
        print("  Columns:")
        for schema in table.schema:
            print(f"    - {schema.name} ({schema.field_type})")
    except Exception as e:
        print(f"  Error inspecting {table_id}: {e}")


def main():
    client = get_client()

    # Key tables to inspect based on the "Reality Check"
    tables_to_check = [
        "cbi-v15.training.daily_ml_matrix",
        "cbi-v15.reference.regime_calendar",
        "cbi-v15.reference.neural_drivers",
        "cbi-v15.reference.train_val_test_splits",
        "cbi-v15.ops.ingestion_completion",
        # Check for existence of proposed tables
        "cbi-v15.meta.feature_catalog",
        "cbi-v15.planning.experiments",
        "cbi-v15.training.runs",
        "cbi-v15.training.predictions",
        "cbi-v15.intelligence.reports",
    ]

    print("=== BIGQUERY SCHEMA INSPECTION ===")

    # 1. Check specific known tables
    for t in tables_to_check:
        inspect_table(client, t)

    # 2. Scan for any other interesting tables in 'training' and 'features' (if exists)
    for dataset_id in ["training", "features", "predictions"]:
        try:
            full_dataset_id = f"{client.project}.{dataset_id}"
            tables = list(client.list_tables(full_dataset_id))
            if tables:
                print(f"\nScanning dataset: {dataset_id} ({len(tables)} tables)")
                for t in tables:
                    if t.table_id not in [x.split(".")[-1] for x in tables_to_check]:
                        # Just list name and row count for others to avoid spam
                        try:
                            table_ref = client.get_table(t)
                            print(f"  - {t.table_id} (Rows: {table_ref.num_rows})")
                        except:
                            print(f"  - {t.table_id}")
        except Exception as e:
            print(f"  Dataset {dataset_id} not found or inaccessible: {e}")


if __name__ == "__main__":
    main()
