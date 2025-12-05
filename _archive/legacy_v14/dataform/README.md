# Dataform: BigQuery ETL Framework

This directory contains all BigQuery ETL transformations using Dataform.

## Overview

Dataform is the primary ETL framework for CBI-V15. All transformations are version controlled, tested, and deployed via CI/CD.

## Structure

```
dataform/
├── dataform.json          # Project configuration
├── includes/              # Shared SQL functions and constants
├── definitions/           # ETL transformations (layered)
│   ├── 01_raw/          # Source declarations
│   ├── 02_staging/      # Cleaned, normalized data
│   ├── 03_features/     # Engineered features
│   ├── 04_training/     # Training-ready tables
│   ├── 05_assertions/   # Data quality gates
│   └── 06_api/          # Public API views
└── schedules/            # Cloud Scheduler configs
```

## Layers

### 01_raw: Source Declarations
External data sources declared here. No transformations, just declarations.

### 02_staging: Cleaned Data
Raw data cleaned, normalized, forward-filled. Incremental MERGE semantics.

### 03_features: Engineered Features
All feature engineering: Big 8 drivers, technical indicators, correlations, etc.

### 04_training: Training Tables
Training-ready tables with targets, horizons, regime weights.

### 05_assertions: Data Quality Gates
Automated data quality checks. Failures block deployment.

### 06_api: Public Views
Dashboard-ready views for Vercel app.

## Usage

### Compile
```bash
cd dataform
dataform compile
```

### Test Assertions
```bash
dataform test
```

### Run All
```bash
dataform run
```

### Run Specific Layer
```bash
dataform run --tags staging
dataform run --tags features
dataform run --tags training
```

## Key Features

- **Incremental ETL**: MERGE semantics with `uniqueKey`
- **Data Quality**: Automated assertions
- **Version Control**: All SQL version controlled
- **CI/CD**: GitHub Actions integration
- **Partitioning**: All tables partitioned by date
- **Clustering**: Optimized for query performance

## Documentation

- [Raw Layer](definitions/01_raw/README.md)
- [Staging Layer](definitions/02_staging/README.md)
- [Features Layer](definitions/03_features/README.md)
- [Training Layer](definitions/04_training/README.md)
- [Assertions](definitions/05_assertions/README.md)
- [API Views](definitions/06_api/README.md)

---

**Last Updated**: November 28, 2025

