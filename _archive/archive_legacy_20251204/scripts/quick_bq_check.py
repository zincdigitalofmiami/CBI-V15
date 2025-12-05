import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from cbi_utils.bigquery_client import get_client


def main():
    try:
        client = get_client()
        print(f"Connected to project: {client.project}")

        print("\nDatasets:")
        datasets = list(client.list_datasets())
        if datasets:
            for dataset in datasets:
                print(f" - {dataset.dataset_id}")

                # List tables in dataset
                tables = list(client.list_tables(dataset.dataset_id))
                if tables:
                    for table in tables:
                        print(f"   - {table.table_id}")
                else:
                    print("   (empty)")
        else:
            print("No datasets found.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
