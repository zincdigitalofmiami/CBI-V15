"""
AnoFox: Build Training Matrix

Adds targets and train/val/test splits to feature matrix.
"""

import os
from pathlib import Path

import duckdb

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")


_NON_FEATURE_COLUMNS = {
    "as_of_date",
    "symbol",
    "train_val_test_split",
    "training_weight",
    "updated_at",
}


def _quote_ident(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _is_numeric_type(data_type: str) -> bool:
    t = data_type.upper()
    return any(
        key in t
        for key in (
            "INT",
            "INTEGER",
            "BIGINT",
            "SMALLINT",
            "TINYINT",
            "HUGEINT",
            "DECIMAL",
            "DOUBLE",
            "REAL",
            "FLOAT",
        )
    )


def build_preconditioning_params(
    con: duckdb.DuckDBPyConnection,
    source_table: str = "training.daily_ml_matrix_zl",
    params_table: str = "training.feature_preconditioning_params_zl",
    train_split_value: str = "train",
    iqr_floor: float = 1e-12,
) -> None:
    """
    Build robust (median/IQR) parameters for each numeric feature, fit on TRAIN split only.

    This prevents leakage and provides a stable, SQL-first way to:
    - Impute missing values (median)
    - Winsorize/clamp outliers (median ± k * IQR)
    - Robust-scale features ((x - median) / IQR)
    """
    cols = con.execute(
        f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'training'
          AND table_name = 'daily_ml_matrix_zl'
        ORDER BY ordinal_position
        """
    ).fetchall()

    feature_cols = [
        name
        for (name, dtype) in cols
        if name not in _NON_FEATURE_COLUMNS
        and not name.startswith("target_")
        and _is_numeric_type(dtype)
    ]

    if not feature_cols:
        raise RuntimeError(f"No numeric feature columns found in {source_table}")

    # UNPIVOT is not available in all DuckDB/MotherDuck builds.
    # Build a long-form table via UNION ALL (safe, explicit).
    split_value = train_split_value.replace("'", "''")
    long_query = "\nUNION ALL\n".join(
        f"""
        SELECT
          '{col.replace("'", "''")}' AS column_name,
          CAST({_quote_ident(col)} AS DOUBLE) AS val
        FROM {source_table}
        WHERE train_val_test_split = '{split_value}'
        """.strip()
        for col in feature_cols
    )

    features_values = ", ".join(
        "('" + col.replace("'", "''") + "')" for col in feature_cols
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {params_table} AS
        WITH feature_list(column_name) AS (
          SELECT column_name FROM (VALUES {features_values}) AS t(column_name)
        ),
        long AS (
          {long_query}
        ),
        stats AS (
          SELECT
            column_name,
            quantile_cont(val, 0.25) AS q25,
            quantile_cont(val, 0.50) AS median,
            quantile_cont(val, 0.75) AS q75
          FROM long
          WHERE val IS NOT NULL
          GROUP BY column_name
        )
        SELECT
          f.column_name,
          COALESCE(s.q25, 0.0) AS q25,
          COALESCE(s.median, 0.0) AS median,
          COALESCE(s.q75, 0.0) AS q75,
          CASE
            WHEN s.column_name IS NULL THEN 1.0
            ELSE GREATEST(s.q75 - s.q25, {float(iqr_floor)})
          END AS iqr,
          CURRENT_TIMESTAMP AS updated_at
        FROM feature_list f
        LEFT JOIN stats s
          ON s.column_name = f.column_name
        """
    )


def build_preconditioned_training_matrix(
    con: duckdb.DuckDBPyConnection,
    source_table: str = "training.daily_ml_matrix_zl",
    params_table: str = "training.feature_preconditioning_params_zl",
    output_table: str = "training.daily_ml_matrix_zl_v15",
    clamp_k: float = 10.0,
) -> None:
    """
    Create a preconditioned training matrix:
    - For every numeric feature column (excluding targets), apply:
      - median imputation
      - winsorization/clamping using IQR (median ± k*IQR)
      - robust scaling: (x - median) / IQR

    Targets and split columns are preserved as-is.
    """
    cols = con.execute(
        f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'training'
          AND table_name = 'daily_ml_matrix_zl'
        ORDER BY ordinal_position
        """
    ).fetchall()

    feature_cols = [
        name
        for (name, dtype) in cols
        if name not in _NON_FEATURE_COLUMNS
        and not name.startswith("target_")
        and _is_numeric_type(dtype)
    ]

    if not feature_cols:
        raise RuntimeError(f"No numeric feature columns found in {source_table}")

    def expr(col: str) -> str:
        col_q = _quote_ident(col)
        # Use scalar subqueries to avoid N-way joins; table is small (thousands of rows).
        median = f"(SELECT median FROM {params_table} WHERE column_name = '{col}')"
        iqr = f"(SELECT iqr FROM {params_table} WHERE column_name = '{col}')"
        raw = f"COALESCE(CAST({col_q} AS DOUBLE), {median})"
        lower = f"({median} - ({clamp_k} * {iqr}))"
        upper = f"({median} + ({clamp_k} * {iqr}))"
        clamped = f"LEAST(GREATEST({raw}, {lower}), {upper})"
        return f"(({clamped} - {median}) / NULLIF({iqr}, 0.0)) AS {col_q}"

    feature_exprs = ",\n            ".join(expr(c) for c in feature_cols)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT
            as_of_date,
            symbol,
            train_val_test_split,
            training_weight,
            {feature_exprs},
            -- Preserve targets as-is
            {", ".join(_quote_ident(c[0]) for c in cols if c[0].startswith("target_"))},
            CURRENT_TIMESTAMP AS updated_at
        FROM {source_table}
        ORDER BY as_of_date
        """
    )


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def _load_local_env() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    _load_dotenv_file(root_dir / ".env")
    _load_dotenv_file(root_dir / ".env.local")


def _iter_motherduck_tokens():
    candidates = [
        ("MOTHERDUCK_TOKEN", os.getenv("MOTHERDUCK_TOKEN")),
        ("motherduck_storage_MOTHERDUCK_TOKEN", os.getenv("motherduck_storage_MOTHERDUCK_TOKEN")),
        ("MOTHERDUCK_READ_SCALING_TOKEN", os.getenv("MOTHERDUCK_READ_SCALING_TOKEN")),
        (
            "motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN",
            os.getenv("motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN"),
        ),
    ]
    for _, value in candidates:
        if not value:
            continue
        token = value.strip().strip('"').strip("'")
        if token.count(".") != 2:
            continue
        yield token


def build_training(con: duckdb.DuckDBPyConnection = None) -> None:
    """
    Build training.daily_ml_matrix_zl from features + targets.
    """
    if con is None:
        _load_local_env()
        for token in _iter_motherduck_tokens():
            try:
                con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={token}")
                con.execute("SELECT 1").fetchone()
                break
            except Exception:
                con = None
        if con is None:
            raise ValueError(
                "MotherDuck token required (set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN)"
            )

    print("Building training matrix...")

    # Create training matrix dynamically to include all features
    con.execute(
        """
        CREATE OR REPLACE TABLE training.daily_ml_matrix_zl AS
        SELECT 
            f.*,
            -- Splits & Weights
            CASE 
                WHEN as_of_date < '2023-01-01' THEN 'train'
                WHEN as_of_date < '2024-01-01' THEN 'val'
                ELSE 'test'
            END AS train_val_test_split,
            1.0 AS training_weight,
            
            -- Targets (forward returns)
            LEAD(close, 5) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_1w,
            LEAD(close, 21) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_1m,
            LEAD(close, 63) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_3m,
            LEAD(close, 126) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_6m,
            LEAD(close, 252) OVER (PARTITION BY symbol ORDER BY as_of_date) / close - 1 AS target_ret_12m
        FROM features.daily_ml_matrix_zl f
    """
    )

    print("Building preconditioning params (train split only)...")
    build_preconditioning_params(con)
    print("Building preconditioned training matrix (V15)...")
    build_preconditioned_training_matrix(con)

    print("Training build complete.")


if __name__ == "__main__":
    build_training()
