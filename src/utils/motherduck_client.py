"""
MotherDuck Client - Centralized Database Connection for Python

Provides a singleton connection to MotherDuck that can be used throughout
the entire project. All Python scripts should import from this module.

This module automatically loads credentials from:
1. Environment variables (MOTHERDUCK_TOKEN, MOTHERDUCK_READ_SCALING_TOKEN, MOTHERDUCK_DB)
2. .env file in project root (via python-dotenv)

Token Priority:
- MOTHERDUCK_TOKEN (read_write) - Primary token for write operations
- motherduck_storage_MOTHERDUCK_TOKEN (from Vercel) - Alternative source
- MOTHERDUCK_READ_SCALING_TOKEN (read_scaling) - For read-only scaling operations

Usage:
    from src.utils.motherduck_client import get_motherduck_connection

    conn = get_motherduck_connection()
    result = conn.execute("SELECT * FROM raw.databento_ohlcv LIMIT 5").fetchall()
"""

import os
from pathlib import Path
from typing import Optional

import duckdb
from dotenv import load_dotenv

# Load .env file from project root (3 levels up from src/utils/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_ENV_LOCAL_PATH = _PROJECT_ROOT / ".env.local"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
if _ENV_LOCAL_PATH.exists():
    load_dotenv(_ENV_LOCAL_PATH)

_connection: Optional[duckdb.DuckDBPyConnection] = None


def _get_motherduck_token() -> str:
    """
    Get MotherDuck token from environment or .env file.

    Checks multiple sources in priority order:
    1. MOTHERDUCK_TOKEN (standard env var)
    2. motherduck_storage_MOTHERDUCK_TOKEN (Vercel storage format)

    Returns:
        MotherDuck token string

    Raises:
        ValueError: If no token is found
    """
    # Priority 1: Vercel storage format (motherduck_storage_MOTHERDUCK_TOKEN)
    #
    # Rationale: we have observed cases where an old/stale MOTHERDUCK_TOKEN exists
    # locally or in CI, but the Vercel-provided storage token is the working one.
    token = os.getenv("motherduck_storage_MOTHERDUCK_TOKEN")
    if token:
        return token.strip().strip('"').strip("'")

    # Priority 2: Standard environment variable
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        return token.strip().strip('"').strip("'")

    raise ValueError(
        "MOTHERDUCK_TOKEN not found. Please set it in:\n"
        "  1. .env file: MOTHERDUCK_TOKEN=your_token\n"
        "  2. Environment: export MOTHERDUCK_TOKEN=your_token\n"
        "  3. Or use: motherduck_storage_MOTHERDUCK_TOKEN (Vercel format)\n"
        f"   Check: {_ENV_PATH}"
    )


def get_motherduck_connection(
    read_only: bool = False, use_read_scaling: bool = False
) -> duckdb.DuckDBPyConnection:
    """
    Get or create a connection to MotherDuck.

    Args:
        read_only: If True, connect in read-only mode
        use_read_scaling: If True, use MOTHERDUCK_READ_SCALING_TOKEN for better read performance

    Returns:
        Active MotherDuck connection

    Raises:
        ValueError: If MOTHERDUCK_TOKEN is not set
    """
    global _connection

    if _connection is not None:
        return _connection

    def _clean_token(value: str) -> str:
        return value.strip().strip('"').strip("'")

    token_candidates = []
    if use_read_scaling:
        token_candidates.extend(
            [
                os.getenv("motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN"),
                os.getenv("MOTHERDUCK_READ_SCALING_TOKEN"),
            ]
        )

    token_candidates.extend(
        [
            os.getenv("motherduck_storage_MOTHERDUCK_TOKEN"),
            os.getenv("MOTHERDUCK_TOKEN"),
        ]
    )

    tokens = []
    for raw in token_candidates:
        if not raw:
            continue
        token = _clean_token(raw)
        if token and token.count(".") == 2:
            tokens.append(token)

    if not tokens:
        tokens = [_get_motherduck_token()]

    db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    last_error: Optional[Exception] = None

    for token in tokens:
        connection_string = f"md:{db_name}?motherduck_token={token}"
        if read_only:
            connection_string += "&access_mode=read_only"
        try:
            con = duckdb.connect(connection_string)
            con.execute("SELECT 1").fetchone()
            _connection = con
            break
        except Exception as e:
            last_error = e
            continue

    if _connection is None:
        raise ValueError(f"Failed to connect to MotherDuck database '{db_name}': {last_error}")

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
    columns = conn.execute(f"DESCRIBE {schema}.{table}").fetchall()

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
