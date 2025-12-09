# Training

## Purpose
Model training code - baseline models, training utilities, validation schemas.

## Structure
```
training/
├── baselines/       # Baseline model implementations
│   └── lightgbm_zl.py
└── utils/           # Training utilities
    └── validation_schema.py
```

## What Belongs Here
- Model training code
- Hyperparameter tuning utilities
- Cross-validation helpers
- Model evaluation code

## What Does NOT Belong Here
- Training configuration (→ `config/training/`)
- Model artifacts (→ cloud storage or `models/`)
- Feature engineering (→ `src/features/`)

## Relationship to TSci
- TSci (TimeSeriesScientist) DECIDES which model to use
- This folder IMPLEMENTS the models
- TSci calls `src/training/baselines/lightgbm_zl.py` to train

## Current Models
- `baselines/lightgbm_zl.py` - LightGBM baseline for ZL (Soybean Oil)

