# Training

> Fast-moving workspace: read `docs/architecture/MASTER_PLAN.md`, `AGENTS.md`, and the active master plan `.cursor/plans/ALL_PHASES_INDEX.md` before editing/adding training code. Avoid duplicate scripts; keep explorer clean.

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

## Relationship to Orchestration
- V15.1: AutoGluon-based training scripts are the canonical modeling path.
- AutoGluon is optional/legacy; it may call these scripts for experimentation, but it is not required for production training.

## Current Models
- `baselines/lightgbm_zl.py` - LightGBM baseline for ZL (Soybean Oil)
- `autogluon/mitra_trainer.py` - Mitra time series foundation model (Metal/MPS accelerated for Mac M4)
- `autogluon/timeseries_trainer.py` - TimeSeriesPredictor wrapper with Mitra fallback

## Big 8 Bucket Modeling Rules

Training code must respect the Big 8 modeling stack:

- Big 8 buckets are: Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility
- Bucket features are provided from SQL macros in `database/macros/` (no Python feature loops)
- Use AutoGluon `TabularPredictor` for all Big 8 bucket specialists
- Use AutoGluon `TimeSeriesPredictor` for core ZL forecasting (or Mitra fallback on Mac M4)
- Meta model fuses Big 8 + core ZL outputs (L0/L1)
- Ensemble layer smooths predictions into final forecasts
- Monte Carlo simulation consumes final forecasts to generate probabilistic scenarios (VaR/CVaR)

## Mitra Integration (Mac M4 Metal Acceleration)

**What**: Salesforce's Mitra time series foundation model, Metal-accelerated via PyTorch MPS backend  
**Where**: `src/training/autogluon/mitra_trainer.py`  
**Why**: Alternative to Chronos-Bolt which hangs on Mac M4 (mutex lock issue)

**Usage**:
```python
from src.training.autogluon.mitra_trainer import MitraForecastWrapper

# Auto-detects Metal (MPS) on Mac M4
mitra = MitraForecastWrapper(
    prediction_length=14,
    device='mps',  # or 'cpu'
    verbosity=1
)

# Zero-shot forecasting (no training required)
forecasts = mitra.predict(time_series_data)
# Returns: DataFrame with columns [timestamp, mean, P10, P50, P90]
```

**Requirements**:
- `mitra-forecast>=0.1.0` (in requirements.txt)
- `torch>=2.0.0` with MPS support (Mac M4)
- macOS 13.0+ (Ventura or newer)

**Performance**:
- ✅ Works well for inference and probabilistic forecasts
- ✅ Metal acceleration provides good performance on Mac M4
- ⚠️  Not suitable for heavy fine-tuning (use AutoGluon TabularPredictor instead)

## Engineering Agent Prompt (Codex/Cursor)

Use this developer prompt when working on training code with Codex/Cursor:

```text
You are the CBI-V15 Engineering Agent.

Follow the system rules and Cursor rules.json.

Task:
I want you to operate strictly within the CBI-V15 architecture.
Before making any changes:
1. Validate context.
2. If any file or directory is missing, ask me for it.
3. Explain your plan BEFORE writing code.
4. Produce minimal, surgical diffs.

Never hallucinate imports, modules, directories, dependencies, or data sources.
Never reintroduce BigQuery or v14 patterns.
Never write code outside the defined directories.
Keep everything aligned with the V15.1 training engine: Big 8 Tabular → Core TS → Meta → Ensemble → Monte Carlo.

When ready, ask: "Show me the files involved in this operation."
```
