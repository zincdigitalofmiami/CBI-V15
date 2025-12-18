# Just recipes for daily workflows

set shell := ["/bin/bash", "-cu"]

# Bootstrap
setup:
    python -m venv .venv
    . .venv/bin/activate && pip install -r config/requirements/requirements.txt
    npm ci
    cd dashboard && npm ci

# Development
dev:
    echo "Starting dashboard..."
    cd dashboard && npm run dev

# Quality gates
qa:
    ruff .
    black --check .
    pytest -q || true
    cd dashboard && npm run lint && npm run typecheck

# Database operations
db:deploy motherduck?=false local?=true:
    if {{local}}; then python scripts/setup_database.py --local; fi
    if {{motherduck}}; then python scripts/setup_database.py --motherduck; fi

db:migrate:
    python database/migrations/migrate.py

db:seed:
    python database/seeds/seed_symbols.py
    python database/seeds/seed_reference_tables.py || true
    python database/seeds/seed_regimes.py || true
    python database/seeds/seed_splits.py

# Sync MotherDuck to local DuckDB (safe: read-only MD)
sync:md-to-local args="--dry-run":
    python scripts/sync_motherduck_to_local.py {{args}}

# Autosave lightweight background commits to a backup branch
# Usage: `just autosave` (Ctrl+C to stop). Commits only when there are changes.
autosave interval_seconds=300 branch="autosave/pr-1":
    echo "Starting autosave to branch '{{branch}}' every {{interval_seconds}}s (Ctrl+C to stop)..."
    while true; do \
        git rev-parse --abbrev-ref HEAD >/dev/null 2>&1 || { echo "Not a git repo"; exit 1; }; \
        current_branch=$(git rev-parse --abbrev-ref HEAD); \
        git fetch -q; \
        git checkout -q -B "{{branch}}" "$current_branch"; \
        git add -A; \
        if ! git diff --cached --quiet; then \
            ts=$(date -u +%FT%TZ); \
            git commit -m "autosave: $ts" >/dev/null; \
            echo "✔ autosaved at $ts"; \
            git push -u -q origin "{{branch}}" || true; \
        fi; \
        git checkout -q "$current_branch"; \
        sleep {{interval_seconds}}; \
    done
