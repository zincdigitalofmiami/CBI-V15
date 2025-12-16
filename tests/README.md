# Tests Directory

## Structure

```
tests/
├── unit/           # Unit tests (fast, isolated)
├── integration/    # Integration tests (database, API)
└── conftest.py     # Pytest fixtures
```

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=src --cov-report=html
```

## Note

Manual verification scripts are in `scripts/`:
- `scripts/test_motherduck_connection.py` - Database connection test
- `scripts/test_scrapecreators_pipeline.py` - ScrapeCreators test
- `scripts/test_autogluon_ts.py` - AutoGluon time series test

