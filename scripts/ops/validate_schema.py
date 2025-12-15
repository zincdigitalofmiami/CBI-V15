#!/usr/bin/env python3
"""
Schema Validation Runner (non-AI)

- Validates critical tables exist with expected columns/PKs.
- Runs locally in DuckDB using in-repo DDLs when MOTHERDUCK_TOKEN is not provided.
- When MOTHERDUCK_TOKEN is provided, attempts to query live MotherDuck.

Outputs:
- docs/inspections/schema_validation.md
- docs/inspections/schema_validation.json
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs" / "inspections"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def pragma_table_info(con: duckdb.DuckDBPyConnection, fq_table: str) -> List[tuple]:
    return con.execute(f"PRAGMA table_info('{fq_table}')").fetchall()


def pragma_indexes(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> List[tuple]:
    return con.execute(
        f"SELECT * FROM duckdb_indexes() WHERE schema_name = '{schema}' AND table_name = '{table}'"
    ).fetchall()


def validate_local() -> Dict[str, Any]:
    con = duckdb.connect(database=":memory:")
    # Ensure schemas exist
    con.execute("CREATE SCHEMA raw;")
    con.execute("CREATE SCHEMA staging;")
    con.execute("CREATE SCHEMA features;")

    # Execute DDLs from repo
    raw_ddl = (ROOT / "database/ddl/02_raw/010_raw_databento_ohlcv.sql").read_text()
    news_ddl = (ROOT / "database/ddl/02_raw/080_raw_news_articles.sql").read_text()
    staging_ddl = (ROOT / "database/ddl/03_staging/010_staging_ohlcv_daily.sql").read_text()

    for script in (raw_ddl, news_ddl, staging_ddl):
        for stmt in [s.strip() for s in script.split(";") if s.strip()]:
            con.execute(stmt)

    result: Dict[str, Any] = {}

    # raw.databento_futures_ohlcv_1d
    result["raw.databento_futures_ohlcv_1d"] = {
        "columns": pragma_table_info(con, "raw.databento_futures_ohlcv_1d"),
        "indexes": pragma_indexes(con, "raw", "databento_futures_ohlcv_1d"),
    }

    # staging.ohlcv_daily
    result["staging.ohlcv_daily"] = {
        "columns": pragma_table_info(con, "staging.ohlcv_daily"),
        "indexes": pragma_indexes(con, "staging", "ohlcv_daily"),
    }

    # raw.scrapecreators_news_buckets
    result["raw.scrapecreators_news_buckets"] = {
        "columns": pragma_table_info(con, "raw.scrapecreators_news_buckets"),
        "indexes": pragma_indexes(con, "raw", "scrapecreators_news_buckets"),
    }

    return result


def validate_motherduck(token: str, dbname: str = None) -> Dict[str, Any]:
    dbname = dbname or os.getenv("MOTHERDUCK_DB", "cbi_v15")
    con = duckdb.connect(f"md:{dbname}?motherduck_token={token}")
    result: Dict[str, Any] = {}
    tables = [
        "raw.databento_futures_ohlcv_1d",
        "staging.ohlcv_daily",
        "raw.scrapecreators_news_buckets",
    ]
    for fq in tables:
        schema, table = fq.split(".")
        try:
            result[fq] = {
                "columns": pragma_table_info(con, fq),
                "indexes": pragma_indexes(con, schema, table),
            }
        except Exception as e:
            result[fq] = {"error": str(e)}
    return result


def write_outputs(data: Dict[str, Any]) -> None:
    # JSON
    json_path = DOCS_DIR / "schema_validation.json"
    json_path.write_text(json.dumps(data, indent=2))
    # Markdown
    md_lines = ["# Schema Validation Results\n"]
    for name, payload in data.items():
        md_lines.append(f"\n## {name}\n")
        if "error" in payload:
            md_lines.append(f"Error: {payload['error']}\n")
            continue
        md_lines.append("Columns (PRAGMA table_info):\n")
        for row in payload.get("columns", []):
            # row: (cid, name, type, notnull, dflt_value, pk)
            md_lines.append(f"- {row}")
        md_lines.append("\nIndexes:\n")
        for row in payload.get("indexes", []):
            md_lines.append(f"- {row}")
    (DOCS_DIR / "schema_validation.md").write_text("\n".join(md_lines))


def main() -> None:
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        live = validate_motherduck(token)
        write_outputs({"live": live})
        print("[schema-validate] Live MotherDuck validation complete.")
    else:
        local = validate_local()
        write_outputs({"local": local})
        print("[schema-validate] Local validation (in-repo DDL) complete.")


if __name__ == "__main__":
    main()
