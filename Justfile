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
    echo "Starting Trigger.dev and Dashboard..."
    (npm run dev &)
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
