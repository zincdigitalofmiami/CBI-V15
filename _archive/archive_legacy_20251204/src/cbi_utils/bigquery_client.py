#!/usr/bin/env python3
"""
BigQuery client utilities for CBI-V15
"""
from google.cloud import bigquery
from typing import Optional, List
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


def merge_staging_to_target(
    staging_table: str,
    target_table: str,
    key_columns: List[str],
    all_columns: List[str],
    project_id: str = PROJECT_ID,
) -> bool:
    """
    Merge rows from a staging table into a target table using BigQuery MERGE.

    Only inserts rows where the key columns do not already exist in the target.

    Args:
        staging_table: Full staging table ID (e.g., "cbi-v15.raw_staging.fred_fx_20251129_010101")
        target_table: Full target table ID (e.g., "cbi-v15.raw.fred_economic")
        key_columns: Columns that form the primary key (e.g., ["series_id", "date"])
        all_columns: All columns to insert (order matters)
        project_id: GCP project ID

    Returns:
        True if MERGE succeeded, False otherwise.
    """
    join_cond = " AND ".join([f"T.{c} = S.{c}" for c in key_columns])
    insert_cols = ", ".join(all_columns)
    insert_vals = ", ".join([f"S.{c}" for c in all_columns])

    query = f"""
MERGE `{target_table}` T
USING `{staging_table}` S
ON {join_cond}
WHEN NOT MATCHED THEN
  INSERT ({insert_cols})
  VALUES ({insert_vals})
"""
    try:
        client = get_client(project_id=project_id)
        logger.info(f"Running MERGE from {staging_table} into {target_table}")
        job = client.query(query)
        job.result()
        logger.info("MERGE completed successfully.")
        return True
    except Exception as e:
        logger.error(f"MERGE failed ({staging_table} -> {target_table}): {e}")
        return False
