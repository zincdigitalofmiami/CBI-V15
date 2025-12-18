#!/usr/bin/env python3
"""
CBI-V15 Schema X-Ray (Diagnostic)

Reads raw schema information from:
  1) CSV source: data/raw/zl_continuous.csv (if present)
  2) MotherDuck via DuckDB: information_schema.columns (if MOTHERDUCK_TOKEN is set)

This script is read-only: it does not create/modify tables or files.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import duckdb
import pandas as pd


@dataclass(frozen=True)
class TableRef:
    schema: str
    table: str

    @property
    def fqtn(self) -> str:
        return f"{self.schema}.{self.table}"


DEFAULT_CSV_PATH = "data/raw/zl_continuous.csv"
DEFAULT_TABLES: List[TableRef] = [
    TableRef("raw", "databento_futures_ohlcv_1d"),
    TableRef("raw", "databento_futures_ohlcv_1h"),
    TableRef("raw", "fred_economic"),
    TableRef("raw", "eia_biofuels"),
    TableRef("raw", "scrapecreators_trump"),
    TableRef("raw", "scrapecreators_news_buckets"),
    TableRef("staging", "market_daily"),
    TableRef("staging", "fred_macro_clean"),
    TableRef("staging", "news_bucketed"),
    TableRef("features", "daily_ml_matrix"),
    TableRef("features", "daily_ml_matrix_zl"),
    TableRef("training", "daily_ml_matrix_zl"),
    TableRef("training", "bucket_predictions"),
]


def _print_header(title: str) -> None:
    print(f"\n--- {title} ---")


def xray_csv(csv_path: str, nrows: int) -> None:
    _print_header("🕵️ SOURCE: CSV")
    if not os.path.exists(csv_path):
        print(f"❌ Not found: {csv_path}")
        return

    print(f"✅ Found: {csv_path}")
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"Rows sampled: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("Pandas dtypes:")
    print(df.dtypes)


def parse_table_refs(table_args: Iterable[str]) -> List[TableRef]:
    out: List[TableRef] = []
    for raw in table_args:
        raw = raw.strip()
        if not raw:
            continue
        if "." not in raw:
            raise ValueError(f"Invalid table ref '{raw}'. Expected format schema.table")
        schema, table = raw.split(".", 1)
        out.append(TableRef(schema=schema, table=table))
    return out


def connect_motherduck() -> duckdb.DuckDBPyConnection:
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise ValueError("MOTHERDUCK_TOKEN environment variable not set")
    db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    return duckdb.connect(f"md:{db_name}?motherduck_token={token}")


def table_exists(conn: duckdb.DuckDBPyConnection, ref: TableRef) -> bool:
    query = """
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
    """
    return conn.execute(query, [ref.schema, ref.table]).fetchone()[0] > 0


def xray_table(conn: duckdb.DuckDBPyConnection, ref: TableRef) -> None:
    if not table_exists(conn, ref):
        print(f"⚠️  Missing table: {ref.fqtn}")
        return

    print(f"✅ Table: {ref.fqtn}")
    cols = conn.execute(
        """
        SELECT
            ordinal_position,
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position
        """,
        [ref.schema, ref.table],
    ).fetchall()

    print(f"Columns ({len(cols)}):")
    for ordinal_position, column_name, data_type, is_nullable in cols:
        print(f"  {ordinal_position:>3}  {column_name}  {data_type}  nullable={is_nullable}")


def xray_motherduck(tables: List[TableRef]) -> None:
    _print_header("🕵️ SOURCE: MOTHERDUCK (DuckDB)")
    try:
        conn = connect_motherduck()
    except Exception as e:
        print(f"❌ Connection unavailable: {e}")
        return

    try:
        for ref in tables:
            xray_table(conn, ref)
    finally:
        conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CBI-V15 schema X-ray (CSV + MotherDuck)")
    p.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to raw CSV to inspect")
    p.add_argument(
        "--csv-nrows",
        type=int,
        default=5,
        help="Number of CSV rows to sample (schema/dtypes only)",
    )
    p.add_argument(
        "--table",
        action="append",
        default=[],
        help="Extra schema.table to inspect (repeatable)",
    )
    p.add_argument(
        "--skip-motherduck",
        action="store_true",
        help="Skip MotherDuck inspection even if token is present",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    print("--- 🕵️ CBI_V15 DATA SCHEMA X-RAY ---")
    xray_csv(args.csv, args.csv_nrows)

    extra_tables = parse_table_refs(args.table)
    tables = DEFAULT_TABLES + extra_tables
    if args.skip_motherduck:
        _print_header("🕵️ SOURCE: MOTHERDUCK (DuckDB)")
        print("⏭️  Skipped by --skip-motherduck")
    else:
        xray_motherduck(tables)

    print("\n--- X-RAY COMPLETE ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

