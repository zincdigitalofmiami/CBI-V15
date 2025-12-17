# Training Config

## Purpose

Model training configuration - hyperparameters, feature lists, split ratios, horizons.

## What Belongs Here

- `model_config.yaml` - Hyperparameters, horizons, splits
- Feature importance weights
- Regime definitions

## What Does NOT Belong Here

- Training code (→ `src/training/`)
- Model artifacts (→ `models/` or cloud storage)
- Training data (→ MotherDuck)

## Naming Convention

`model_config.yaml` is the canonical file. Horizon-specific: `{horizon}_config.yaml`
