#!/usr/bin/env python3
"""
BigQuery client utilities for CBI-V15
"""
from google.cloud import bigquery
from typing import Optional
import logging

logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
LOCATION = "us-central1"

def get_client(project_id: str = PROJECT_ID, location: str = LOCATION) -> bigquery.Client:
    """
    Get BigQuery client
    
    Args:
        project_id: GCP project ID
        location: BigQuery location
        
    Returns:
        BigQuery client instance
    """
    client = bigquery.Client(project=project_id, location=location)
    return client

def query_to_dataframe(query: str, project_id: str = PROJECT_ID) -> Optional:
    """
    Execute BigQuery query and return as DataFrame
    
    Args:
        query: SQL query string
        project_id: GCP project ID
        
    Returns:
        pandas DataFrame or None on error
    """
    try:
        client = get_client(project_id)
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return None

def load_dataframe_to_table(
    df,
    table_id: str,
    project_id: str = PROJECT_ID,
    write_disposition: str = "WRITE_APPEND"
) -> bool:
    """
    Load pandas DataFrame to BigQuery table
    
    Args:
        df: pandas DataFrame
        table_id: Full table ID (e.g., "cbi-v15.raw.databento_futures_ohlcv_1d")
        project_id: GCP project ID
        write_disposition: WRITE_APPEND, WRITE_TRUNCATE, or WRITE_EMPTY
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_client(project_id)
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition
        )
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for job to complete
        logger.info(f"Loaded {len(df)} rows to {table_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to load to {table_id}: {e}")
        return False

