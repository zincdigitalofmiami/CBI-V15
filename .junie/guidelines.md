### CBI‑V15 — Development Guidelines (Project‑Specific)

These notes capture how to reliably set up, test, and work on this repo on a Mac (primary target), with specifics for MotherDuck/DuckDB, Trigger.dev jobs, and the Next.js dashboard. They assume an experienced developer; only project‑specific details are included.

#### Build and Configuration

- Python runtime: 3.12+ (virtualenv recommended). All Python deps are pinned in `config/requirements/requirements.txt`.
  - Quick bootstrap (uses `Justfile`):
    - `just setup` → creates `.venv`, installs Python deps, runs `npm ci` at root and in `dashboard/`.
  - Manual bootstrap:
    - `python -m venv .venv && source .venv/bin/activate`
    - `pip install -r config/requirements/requirements.txt`
    - `npm ci && (cd dashboard && npm ci)`

- Node: 20.17.0 via Volta (see `package.json` → `volta` block). If you use Volta, it will pin Node/NPM automatically; otherwise install Node 20.x.

- Databases: MotherDuck (cloud) is the source of truth; a local DuckDB mirror is used for fast I/O.
  - Env vars used by scripts:
    - `MOTHERDUCK_TOKEN` — required for any MotherDuck access.
    - `MOTHERDUCK_DB` — logical DB name (default `cbi_v15`).
  - Creating schemas/tables/macros:
    - Local only: `python scripts/setup_database.py --local`
    - MotherDuck only: `python scripts/setup_database.py --motherduck`
    - Both: `python scripts/setup_database.py --both`
    - Add `--force` to drop and recreate all schemas. Macros are loaded unless `--skip-macros` is given.
  - Local DB path: `data/duckdb/cbi_v15.duckdb` (created automatically by scripts).

- Databento: Ingestion jobs use the official SDK. API key should be in your environment (e.g., `.env` or shell profile) as `DATABENTO_API_KEY`. See `trigger/DataBento/Scripts/collect_daily.py` and project audit docs for context.

- Dashboard: Next.js 14 app in `dashboard/` connects to MotherDuck via DuckDB‑WASM. If you see empty charts, double‑check the queried table names match the ingestion outputs (see audit notes in `DATABENTO_INGESTION_AUDIT.md`).

#### Testing

- Python tests are collected from `tests/` (see `pytest.ini`). The repository also includes SQL tests under `database/tests/sql/` that can be executed through a harness or a smoke‑test runner.

- Installing test tooling: already covered by `pip install -r config/requirements/requirements.txt` (includes `pytest`, `pytest-cov`, `duckdb`).

- Running Python tests (selective): because some tests depend on the full SQL macro stack and DB contents, prefer targeted runs during local development.
  - Example: `pytest -q -k sanity`
    - We validated this pattern by creating and running a trivial `tests/test_sanity.py` locally, then removing it. The command executed successfully: “1 passed, N deselected”. Use `-k` to scope to safe tests when the DB isn’t initialized.
  - Full run: `pytest -q`
    - Expect failures if the database schemas/macros aren’t loaded or if the local DuckDB mirror is missing.

- SQL smoke tests: `scripts/sql_smoke_tests.py`
  - Local DuckDB (no MotherDuck token required):
    - `python scripts/sql_smoke_tests.py`
      - If `database/tests/sql/` contains `test_*.sql`, the harness at `database/tests/harness.py` runs them against your current connection.
      - If not, a built‑in quick check runs and will look for `forecasts.zl_predictions` and basic freshness. On a pristine DB this will WARN/FAIL, which is expected until you deploy schemas and load data.
  - MotherDuck (requires token):
    - `MOTHERDUCK_TOKEN=… MOTHERDUCK_DB=cbi_v15 python scripts/sql_smoke_tests.py --motherduck`
  - To prepare a clean local DB for tests: `python scripts/setup_database.py --local --force`

- Adding new tests:
  - Python: create files under `tests/` with `test_*.py`. Prefer pure‑logic unit tests (no network). For DB‑dependent tests, ensure `scripts/setup_database.py` has been run and consider marking them, e.g., `@pytest.mark.db` and use `-m db` gates.
  - SQL: add `test_*.sql` under `database/tests/sql/`. The harness expects either a `test_result` column (values: `PASS`/`FAIL`/`WARN`/`SKIP`) or will print raw results as `INFO`.

#### Development Tips and Conventions

- Code Style
  - Python: Use `ruff` and `black`. `just qa` runs `ruff .`, `black --check .`, and `pytest -q`. Follow existing naming and module layout under `src/` and `scripts/`.
  - TypeScript/JS (dashboard + trigger jobs): use ESLint/Prettier; `npm run lint`, `npm run typecheck`, `npm run format`. Node 20.17.0 recommended.

- Data Warehouse Lifecycle
  - Always run `scripts/setup_database.py` after making changes under `database/ddl/` or `database/macros/`.
  - Use `scripts/sync_motherduck_to_local.py` to mirror MotherDuck → local DuckDB for fast training I/O.
  - For destructive resets during local dev: `python scripts/setup_database.py --local --force`.

- Environment Hygiene
  - Keep `MOTHERDUCK_TOKEN` out of code. Use shell env or a private `.env.local` that is not committed.
  - The dashboard queries must align with ingestion outputs. As of the latest audit, ingestion writes to `raw.databento_futures_ohlcv_1d`; ensure dashboard queries the same when displaying historical prices.

- Orchestration
  - Trigger.dev CLI is installed via devDependencies. Use `npm run dev` at repo root for jobs, and `cd dashboard && npm run dev` for the dashboard. The `just dev` recipe will launch both (root first in background, dashboard in foreground).

- CI/Quality Gates
  - `just qa` executes an opinionated local CI subset. Fix lint and type errors before pushing. Some tests are integration‑level and will require a prepared DB.

#### Verified Commands (executed locally during guideline creation)

- Python unit test (scoped):
  - `pytest -q -k sanity` → Passed with deselected others.

- Notes:
  - We intentionally used a scoped pattern to avoid DB‑dependent tests in environments where MotherDuck/DuckDB weren’t initialized. For end‑to‑end verification, set up the DB first, then run SQL smoke tests or the full pytest suite.
