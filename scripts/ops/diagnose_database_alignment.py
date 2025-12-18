#!/usr/bin/env python3
"""
Diagnose database-name alignment across:
- MotherDuck (actual databases that exist)
- Local config (.env / env vars)
- Vercel env vars (via `vercel env ls`, if available)
- GitHub Actions workflow defaults (repo file inspection)

Goal: prevent "split brain" where different environments write/read different DBs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import duckdb
    from dotenv import load_dotenv
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\nInstall: pip install duckdb python-dotenv"
    ) from e


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "data_ingestion.yml"

CANONICAL_DB = "cbi_v15"
SUSPECT_VERCEL_DB_LABEL = "motherduck-cbi-v15"


def _mask(value: str) -> str:
    v = value.strip().strip('"').strip("'")
    if len(v) <= 12:
        return "***"
    return f"{v[:8]}...{v[-4:]}"


def _load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def _get_token() -> Tuple[str, str]:
    """
    Token precedence mirrors our repo conventions:
    1) MOTHERDUCK_TOKEN
    2) motherduck_storage_MOTHERDUCK_TOKEN (Vercel integration name)
    """
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        return token.strip().strip('"').strip("'"), "MOTHERDUCK_TOKEN"
    token = os.getenv("motherduck_storage_MOTHERDUCK_TOKEN")
    if token:
        return token.strip().strip('"').strip("'"), "motherduck_storage_MOTHERDUCK_TOKEN"
    raise ValueError(
        "No MotherDuck token found. Set MOTHERDUCK_TOKEN (preferred) or "
        "motherduck_storage_MOTHERDUCK_TOKEN."
    )


def _connect(db: str, token: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(f"md:{db}?motherduck_token={token}")


def _db_summary(db: str, token: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"db": db, "ok": False, "error": None}
    try:
        con = _connect(db, token)
        schemas = [r[0] for r in con.execute(
            "SELECT schema_name FROM information_schema.schemata ORDER BY 1"
        ).fetchall()]
        tables_total = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
        ).fetchone()[0]
        tables_by_schema = con.execute(
            """
            SELECT table_schema, COUNT(*) AS n_tables
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
            GROUP BY 1
            ORDER BY 2 DESC, 1
            """
        ).fetchall()
        con.close()
        out.update(
            {
                "ok": True,
                "schemas": schemas,
                "tables_total": tables_total,
                "tables_by_schema": tables_by_schema,
            }
        )
    except Exception as e:
        out["error"] = str(e)
    return out


def _list_account_dbs(token: str) -> List[Tuple[Any, ...]]:
    con = duckdb.connect(f"md:?motherduck_token={token}")
    rows = con.execute("PRAGMA database_list").fetchall()
    con.close()
    return rows


def _try_vercel_env_ls() -> Optional[str]:
    if not shutil.which("vercel"):
        return None
    try:
        # `vercel env ls` needs to run inside the linked project directory.
        res = subprocess.run(
            ["vercel", "env", "ls"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    except Exception:
        return None


def _workflow_mentions_db() -> Optional[str]:
    if not WORKFLOW_PATH.exists():
        return None
    text = WORKFLOW_PATH.read_text(errors="replace")
    # Keep this simple: we only report presence/lines, not parse YAML.
    hits: List[str] = []
    for line in text.splitlines():
        if "MOTHERDUCK_DB" in line:
            hits.append(line.strip())
    return "\n".join(hits) if hits else ""


def main() -> int:
    _load_env()

    print("=" * 80)
    print("DATABASE ALIGNMENT DIAGNOSTIC")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Canonical DB: {CANONICAL_DB}")
    print("")

    # Local env view
    local_db = os.getenv("MOTHERDUCK_DB", CANONICAL_DB).strip().strip('"').strip("'")
    print("[Local configuration]")
    print(f"- .env present: {'yes' if ENV_PATH.exists() else 'no'} ({ENV_PATH})")
    print(f"- MOTHERDUCK_DB: {local_db!r} (default {CANONICAL_DB!r})")

    try:
        token, token_source = _get_token()
    except Exception as e:
        print(f"- Token: ❌ {e}")
        print("\nFix: set MOTHERDUCK_TOKEN in .env or your shell.")
        return 2

    print(f"- Token source: {token_source} ({_mask(token)})")
    print("")

    # MotherDuck account reality
    print("[MotherDuck account: databases that actually exist]")
    try:
        dbs = _list_account_dbs(token)
        print(f"- PRAGMA database_list returned {len(dbs)} entries:")
        for row in dbs:
            print(f"  - {row}")
    except Exception as e:
        print(f"- ❌ Could not list databases: {e}")
        return 2
    print("")

    # Probe both names
    candidates = [CANONICAL_DB, SUSPECT_VERCEL_DB_LABEL]
    results = {db: _db_summary(db, token) for db in candidates}

    print("[Database probes]")
    for db in candidates:
        r = results[db]
        if r["ok"]:
            print(f"- ✅ {db}: {r['tables_total']} tables, {len(r['schemas'])} schemas")
            top = r["tables_by_schema"][:8]
            if top:
                print("  Top schemas by table count:")
                for schema, n_tables in top:
                    print(f"    - {schema}: {n_tables}")
        else:
            print(f"- ❌ {db}: {r['error']}")
    print("")

    # Vercel env vars (best-effort)
    print("[Vercel env vars (best-effort)]")
    vercel_out = _try_vercel_env_ls()
    if vercel_out is None:
        print("- vercel CLI not available; skipping")
    else:
        has_db = "MOTHERDUCK_DB" in vercel_out
        print(f"- `vercel env ls` contains MOTHERDUCK_DB: {'yes' if has_db else 'no'}")
        if not has_db:
            print(f"- Expected value: {CANONICAL_DB!r} for Production/Preview/Development")
    print("")

    # GitHub Actions workflow (repo inspection)
    print("[GitHub Actions defaults]")
    workflow_hits = _workflow_mentions_db()
    if workflow_hits is None:
        print(f"- Workflow not found: {WORKFLOW_PATH}")
    elif workflow_hits == "":
        print(f"- No MOTHERDUCK_DB lines found in: {WORKFLOW_PATH}")
    else:
        print(f"- Found MOTHERDUCK_DB lines in {WORKFLOW_PATH}:")
        for line in workflow_hits.splitlines():
            print(f"  - {line}")
    print("")

    # Verdict
    print("[Verdict]")
    canonical_ok = results[CANONICAL_DB]["ok"]
    suspect_ok = results[SUSPECT_VERCEL_DB_LABEL]["ok"]

    if not canonical_ok:
        print(f"- ❌ Canonical DB {CANONICAL_DB!r} is not accessible. Stop and fix tokens/db.")
        return 2

    if local_db != CANONICAL_DB:
        print(
            f"- ⚠️ Local MOTHERDUCK_DB is {local_db!r} but canonical is {CANONICAL_DB!r}."
        )

    if suspect_ok:
        print(
            f"- ⚠️ {SUSPECT_VERCEL_DB_LABEL!r} exists in MotherDuck. This can cause split-brain "
            "if any environment points at it."
        )
    else:
        print(
            f"- ✅ {SUSPECT_VERCEL_DB_LABEL!r} does NOT exist in MotherDuck (good: less split-brain risk)."
        )

    if local_db == CANONICAL_DB and canonical_ok and not suspect_ok:
        print("- ✅ Alignment looks good locally. Next: ensure Vercel sets MOTHERDUCK_DB=cbi_v15.")
        return 0

    print("- ⚠️ Alignment is not guaranteed. Fix env vars to use canonical DB everywhere.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
