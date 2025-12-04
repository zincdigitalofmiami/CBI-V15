# Database

This directory contains all SQL schema definitions and macros for the V15 system.

## Structure

```text
database/
├── schema/             # DDL files (run in order)
│   ├── 00_init.sql     # Create schemas, install extensions
│   ├── 06_reference_tables.sql  # feature_catalog, model_registry
│   └── 07_tsci_tables.sql       # jobs, runs, qa_checks
└── macros/             # AnoFox SQL macros
    └── features.sql    # Big 8 feature logic
```

## Schemas

| Schema     | Owner   | Purpose                          |
|------------|---------|----------------------------------|
| raw        | AnoFox  | Immutable source data            |
| staging    | AnoFox  | Intermediate transformations     |
| features   | AnoFox  | Feature tables                   |
| training   | AnoFox  | ML training matrices             |
| forecasts  | AnoFox  | Model predictions                |
| reference  | AnoFox  | Catalogs and registries          |
| ops        | AnoFox  | Ingestion/QA status              |
| tsci       | TSci    | Agent jobs, runs, QA checks      |

## V15 Contract

- AnoFox owns all schemas except `tsci.*`.
- TSci reads from `training.*` and `forecasts.*`, writes only to `tsci.*`.
