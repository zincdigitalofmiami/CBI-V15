#!/usr/bin/env python3
"""
Create BigQuery datasets for CBI-V15
All datasets in us-central1 only
"""
from google.cloud import bigquery
import logging

logging.basicConfig(level=logging.INFO)
client = bigquery.Client(project="cbi-v15")

DATASETS = [
    {
        "dataset_id": "raw",
        "description": "Raw source data declarations",
        "location": "us-central1"
    },
    {
        "dataset_id": "staging",
        "description": "Cleaned, normalized data",
        "location": "us-central1"
    },
    {
        "dataset_id": "features",
        "description": "Engineered features",
        "location": "us-central1"
    },
    {
        "dataset_id": "training",
        "description": "Training-ready tables with targets",
        "location": "us-central1"
    },
    {
        "dataset_id": "forecasts",
        "description": "Model predictions",
        "location": "us-central1"
    },
    {
        "dataset_id": "api",
        "description": "Public API views for dashboard",
        "location": "us-central1"
    },
    {
        "dataset_id": "reference",
        "description": "Reference tables and mappings",
        "location": "us-central1"
    },
    {
        "dataset_id": "ops",
        "description": "Operations monitoring and audit",
        "location": "us-central1"
    }
]

def create_datasets():
    """Create all BigQuery datasets"""
    for dataset_config in DATASETS:
        dataset_id = dataset_config["dataset_id"]
        dataset_ref = client.dataset(dataset_id)
        
        # Check if dataset exists
        try:
            client.get_dataset(dataset_ref)
            logging.info(f"✅ Dataset '{dataset_id}' already exists")
        except Exception:
            # Create dataset
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = dataset_config["location"]
            dataset.description = dataset_config["description"]
            
            dataset = client.create_dataset(dataset, exists_ok=True)
            logging.info(f"✅ Created dataset '{dataset_id}' in {dataset_config['location']}")

if __name__ == "__main__":
    logging.info("Creating BigQuery datasets for CBI-V15...")
    create_datasets()
    logging.info("✅ All datasets created successfully")

