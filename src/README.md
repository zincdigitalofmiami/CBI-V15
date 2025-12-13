# Source Code (src/)

## Purpose
Core Python application code. This is where the business logic lives.

## Structure
```
src/
├── engines/       # Model engine registry (AutoGluon + legacy baselines like LightGBM/TFT)
├── features/      # Feature engineering code
├── ingestion/     # Data collectors (one folder per source)
├── models/        # Model definitions (if needed locally)
├── training/      # Training pipelines
└── utils/         # Shared utilities
```

## What Belongs Here
- Python modules that are imported by other code
- Business logic
- Data collectors
- ML pipelines

## What Does NOT Belong Here
- Operational scripts (→ `scripts/`)
- Configuration (→ `config/`)
- SQL definitions (→ `database/`)

## Naming Convention
- Folders: `snake_case`
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
