"""
MotherDuck Client - Centralized Database Connection for Python

Provides a singleton connection to MotherDuck that can be used throughout
the entire project. All Python scripts should import from this module.

Usage:
    from src.utils.motherduck_client import get_motherduck_connection

    conn = get_motherduck_connection()
    result = conn.execute("SELECT * FROM raw.databento_ohlcv LIMIT 5").fetchall()
"""

import os
import duckdb
from typing import Optional

_connection: Optional[duckdb.DuckDBPyConnection] = None


def get_motherduck_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Get or create a connection to MotherDuck.

    Args:
        read_only: If True, connect in read-only mode

    Returns:
        Active MotherDuck connection

    Raises:
        ValueError: If MOTHERDUCK_TOKEN is not set
    """
    global _connection

    if _connection is not None:
        return _connection

    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        raise ValueError(
            "MOTHERDUCK_TOKEN environment variable not set. "
            "Please set it in your .env file or environment."
        )

    db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    connection_string = f"md:{db_name}?motherduck_token={motherduck_token}"

    if read_only:
        connection_string += "&access_mode=read_only"

    _connection = duckdb.connect(connection_string)

    # Set memory limit for better performance
    _connection.execute("SET memory_limit='8GB'")
    _connection.execute("SET threads=4")

    return _connection


def close_motherduck_connection() -> None:
    """
    Close the active MotherDuck connection.
    Should be called at the end of long-running scripts.
    """
    global _connection

    if _connection is not None:
        _connection.close()
        _connection = None


def execute_query(sql: str, params: Optional[tuple] = None):
    """
    Execute a SQL query on MotherDuck.

    Args:
        sql: SQL query to execute
        params: Optional tuple of parameters for parameterized queries

    Returns:
        Query result
    """
    conn = get_motherduck_connection()

    if params:
        return conn.execute(sql, params)
    else:
        return conn.execute(sql)


def get_table_info(schema: str, table: str) -> dict:
    """
    Get information about a table in MotherDuck.

    Args:
        schema: Schema name (e.g., 'raw', 'staging', 'gold')
        table: Table name

    Returns:
        Dictionary with table metadata
    """
    conn = get_motherduck_connection()

    # Get row count
    row_count = conn.execute(
        f"SELECT COUNT(*) as cnt FROM {schema}.{table}"
    ).fetchone()[0]

    # Get column info
    columns = conn.execute(
        f"DESCRIBE {schema}.{table}"
    ).fetchall()

    return {
        "schema": schema,
        "table": table,
        "row_count": row_count,
        "columns": [{"name": col[0], "type": col[1]} for col in columns],
    }


# Convenience functions for common operations
def insert_dataframe(df, table: str, schema: str = "staging") -> None:
    """
    Insert a pandas DataFrame into a MotherDuck table.

    Args:
        df: Pandas DataFrame to insert
        table: Target table name
        schema: Target schema (default: 'staging')
    """
    conn = get_motherduck_connection()
    conn.execute(f"INSERT INTO {schema}.{table} SELECT * FROM df")


def fetch_dataframe(sql: str):
    """
    Execute a SQL query and return results as a pandas DataFrame.

    Args:
        sql: SQL query to execute

    Returns:
        Pandas DataFrame with query results
    """
    conn = get_motherduck_connection()
    return conn.execute(sql).df()
