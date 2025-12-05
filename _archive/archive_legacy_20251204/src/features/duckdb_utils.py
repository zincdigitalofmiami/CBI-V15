"""
DuckDB Utilities for Feature Engineering
Provides helper functions to read from DuckDB instead of BigQuery.
"""

import duckdb
from pathlib import Path
from typing import Optional
import pandas as pd

DUCKDB_PATH = Path('/Volumes/Satechi Hub/ZL-Intelligence/duckdb/cbi-v15.duckdb')

# Global connection (lazy initialization)
_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Get or create DuckDB connection."""
    global _conn
    if _conn is None:
        if not DUCKDB_PATH.exists():
            raise FileNotFoundError(f"DuckDB database not found at {DUCKDB_PATH}")
        _conn = duckdb.connect(str(DUCKDB_PATH))
    return _conn


def load_table_from_duckdb(schema: str, table: str, columns: Optional[list] = None, where: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from DuckDB table.
    
    Args:
        schema: Schema name (e.g., 'raw', 'staging', 'features')
        table: Table name
        columns: Optional list of columns to select (None = all columns)
        where: Optional WHERE clause (without the WHERE keyword)
    
    Returns:
        pandas DataFrame
    """
    conn = get_duckdb_connection()
    
    # Build SELECT clause
    if columns:
        cols_str = ', '.join(columns)
    else:
        cols_str = '*'
    
    # Build query
    query = f"SELECT {cols_str} FROM {schema}.{table}"
    
    if where:
        query += f" WHERE {where}"
    
    return conn.execute(query).df()


def load_table_sql(sql: str) -> pd.DataFrame:
    """
    Execute SQL query against DuckDB and return DataFrame.
    
    Args:
        sql: SQL query string
    
    Returns:
        pandas DataFrame
    """
    conn = get_duckdb_connection()
    return conn.execute(sql).df()


def close_connection():
    """Close DuckDB connection."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None

