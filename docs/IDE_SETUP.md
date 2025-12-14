# IDE Setup (IntelliJ IDEA Ultimate)

This guide sets up IntelliJ IDEA Ultimate as the single IDE for Python + SQL + JS/TS for CBI-V15.

## 1) Required plugins
- Python
- Jupyter
- Database Tools and SQL (DataGrip)
- JavaScript and TypeScript
- Tailwind CSS
- Docker
- Markdown
- GitHub
- JetBrains AI Assistant

All are bundled with IDEA Ultimate, except AI Assistant (licensed add-on).

## 2) Interpreters and SDKs
- Python: point IDEA to the project venv (.venv) or Conda env.
- Node: use the project Node runtime (Volta/asdf recommended) and enable "Use project node interpreter".

## 3) Environment variables
Create .env files (not committed) or use macOS Keychain + direnv. Minimum:
- MOTHERDUCK_DB
- MOTHERDUCK_TOKEN
- Data source keys used by Trigger.dev (e.g., DATABENTO_API_KEY, FRED_API_KEY)

## 4) Run Configurations
Already provided under .idea/runConfigurations/:
- Dev - Trigger.dev: runs `trigger.dev dev` at repo root
- Dev - Dashboard: runs `next dev` inside dashboard/
- Dev - All: compound that launches both
- QA - All: shell task that runs Python + JS quality checks

Open Run/Debug Configurations and duplicate/tweak as needed. Provide env vars via the configuration Environment tab.

## 5) Databases (DuckDB + MotherDuck)
- Add DuckDB data source pointed at data/duckdb/cbi_v15.duckdb.
- Add MotherDuck via DuckDB driver using your MOTHERDUCK_TOKEN.
- Save data sources in local IDE settings (do not commit secrets).

## 6) Jupyter & Scientific
- Enable Scientific Mode (View -> Appearance -> Scientific Mode).
- Use the project interpreter for notebooks; variables and plots appear in the scientific tool window.

## 7) AI Assistant workflow
- Use chat to explain unfamiliar files (database/macros/*, trigger/*).
- Ask for quick test drafts or refactors. Use AI Diff Review on PRs.

## 8) Optional quality gates
- Add pre-commit (Python) and husky/lint-staged (JS/TS) locally for auto-fix on commit.
- In CI: run ESLint/TS typecheck, Ruff/Black, Pytest, Qodana.

---

Tips
- Use a Compound configuration for one-click dev (Dev - All).
- The HTTP Client (.http files) is great for exercising Trigger.dev webhooks.
- Database tool window: pin consoles to schemas for faster macro iteration.