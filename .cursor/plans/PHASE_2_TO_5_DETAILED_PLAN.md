# CBI-V15.1 Implementation Plan: Phase 2 → Phase 5 (Continued)

**This document continues from `.cursor/plans/PHASE_0_TO_5_DETAILED_PLAN.md`**

---

## 🤖 PHASE 2: AUTOGLUON INTEGRATION

**Goal:** Integrate AutoGluon 1.4 with TabularPredictor and TimeSeriesPredictor wrappers  
**Status:** NOT STARTED  
**Dependencies:** Phase 0 complete, Phase 1 complete

### Task 2.1: Create Mac M4 AutoGluon Setup Script (HIGH RISK)
**UUID:** `nmDVCuPjvqA2A9XRA6YD7J`

**Purpose:** Install AutoGluon 1.4 with Mac M4 compatibility (libomp fix)

**File:** `scripts/setup/install_autogluon_mac.sh`

**Implementation:**
```bash
#!/bin/bash
set -e

echo "Installing AutoGluon 1.4 for Mac M4..."

# 1. Install libomp via Homebrew
brew install libomp

# 2. Set environment variables for libomp
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# 3. Install AutoGluon
pip install 'autogluon.tabular[all]>=1.4.0'
pip install 'autogluon.timeseries[all]>=1.4.0'

# 4. Verify installation
python -c 'from autogluon.tabular import TabularPredictor; print("✓ TabularPredictor ready")'
python -c 'from autogluon.timeseries import TimeSeriesPredictor; print("✓ TimeSeriesPredictor ready")'

echo "✅ AutoGluon 1.4 installed successfully"
```

**Validation:**
```bash
bash scripts/setup/install_autogluon_mac.sh
python -c 'from autogluon.tabular import TabularPredictor; from autogluon.timeseries import TimeSeriesPredictor; print("AutoGluon 1.4 ready")'
# Expected: No import errors, foundation models available
```

**Risk:** HIGH - Mac M4 may have libomp compatibility issues

---

### Task 2.2: Create src/training/autogluon/ Directory Structure (LOW RISK)
**UUID:** `vxqqByVLHr9KUe1KB83y8M`

**Files to Create:**
```bash
mkdir -p src/training/autogluon
touch src/training/autogluon/__init__.py
touch src/training/autogluon/tabular_trainer.py
touch src/training/autogluon/timeseries_trainer.py
touch src/training/autogluon/bucket_specialist.py
touch src/training/autogluon/foundation_models.py
touch src/training/autogluon/ensemble_combiner.py
```

**Validation:**
```bash
ls -la src/training/autogluon/
# Expected: 6 Python files created
```

---

### Task 2.3: Create TabularPredictor Wrapper with Quantile Regression (MEDIUM RISK)
**UUID:** `vuAnAqaTPqVFmUr48UQ1xr`

**File:** `src/training/autogluon/tabular_trainer.py`

**Purpose:** Wrapper for AutoGluon TabularPredictor with quantile regression

**Implementation:**
```python
from autogluon.tabular import TabularPredictor
import pandas as pd
from pathlib import Path

def train_tabular(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    preset: str = 'best_quality',
    quantiles: list = [0.1, 0.5, 0.9],
    time_limit: int = 3600,
    model_path: str = None
):
    """
    Train AutoGluon TabularPredictor with quantile regression.
    
    Args:
        train_df: Training data
        val_df: Validation data
        target_col: Target column name
        preset: 'best_quality' for main model, 'extreme' for bucket specialists
        quantiles: Quantile levels for probabilistic forecasts
        time_limit: Training time limit in seconds
        model_path: Path to save model (default: data/models/{target_col}/)
    
    Returns:
        predictor: Trained AutoGluon TabularPredictor
    """
    if model_path is None:
        model_path = f"data/models/{target_col}"
    
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    predictor = TabularPredictor(
        label=target_col,
        problem_type='quantile',
        quantile_levels=quantiles,
        path=model_path
    )
    
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        presets=preset,
        time_limit=time_limit,
        num_stack_levels=1,  # Creates L1 + WeightedEnsemble_L2
        num_bag_folds=8
    )
    
    return predictor
```

**Validation:**
```bash
python -c 'from src.training.autogluon.tabular_trainer import train_tabular; print("Wrapper ready")'
# Expected: No import errors
```

---

### Task 2.4: Create TimeSeriesPredictor Wrapper with Chronos-Bolt (MEDIUM RISK)
**UUID:** `efED7rpvbH4E4ejX9KVdcS`

**File:** `src/training/autogluon/timeseries_trainer.py`

**Purpose:** Wrapper for AutoGluon TimeSeriesPredictor with Chronos-Bolt

**Implementation:**
```python
from autogluon.timeseries import TimeSeriesPredictor
import pandas as pd

def train_timeseries(
    train_df: pd.DataFrame,
    target_col: str,
    known_covariates: list = None,
    prediction_length: int = 5,
    model_path: str = None
):
    """
    Train AutoGluon TimeSeriesPredictor with Chronos-Bolt.
    
    Args:
        train_df: Training data (time series format)
        target_col: Target column name
        known_covariates: List of known covariate columns (Big 8 bucket scores)
        prediction_length: Forecast horizon (days)
        model_path: Path to save model
    
    Returns:
        predictor: Trained TimeSeriesPredictor
    """
    if model_path is None:
        model_path = f"data/models/timeseries_{target_col}"
    
    predictor = TimeSeriesPredictor(
        target=target_col,
        known_covariates_names=known_covariates,
        prediction_length=prediction_length,
        path=model_path,
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    predictor.fit(
        train_data=train_df,
        hyperparameters={
            'Chronos': {},  # Chronos-Bolt zero-shot baseline
            'PerStepTabular': {'model': 'CatBoost'}  # For comparison
        },
        time_limit=3600
    )
    
    return predictor
```

**Validation:**
```bash
python -c 'from src.training.autogluon.timeseries_trainer import train_timeseries; print("Wrapper ready")'
# Expected: No import errors, Chronos-Bolt available
```

---

### Task 2.5: Create Foundation Models Configuration (LOW RISK)
**UUID:** `chRBL6LYNgCEac3Zu5D8Gc`

**File:** `src/training/autogluon/foundation_models.py`

**Implementation:**
```python
"""
Foundation models configuration for AutoGluon 1.4.
All models run on CPU (Mac M4), no GPU required.
"""

FOUNDATION_MODELS = {
    'Mitra': {
        'name': 'Mitra',
        'max_samples': 10000,
        'use_gpu': False,
        'description': 'Foundation model for tabular data'
    },
    'TabPFNv2': {
        'name': 'TabPFNv2',
        'max_samples': 10000,
        'use_gpu': False,
        'description': 'Prior-fitted network for tabular data'
    },
    'TabICL': {
        'name': 'TabICL',
        'max_samples': 10000,
        'use_gpu': False,
        'description': 'In-context learning for tabular data'
    },
    'Chronos': {
        'name': 'Chronos',
        'model_path': 'amazon/chronos-bolt-small',
        'description': 'Zero-shot time series forecasting'
    }
}

def get_foundation_model_config(model_name: str) -> dict:
    """Get configuration for a specific foundation model."""
    return FOUNDATION_MODELS.get(model_name, {})
```

**Validation:**
```bash
python -c 'from src.training.autogluon.foundation_models import FOUNDATION_MODELS; print(len(FOUNDATION_MODELS))'
# Expected: 4 foundation models configured
```

---

### Task 2.6: Update engine_registry.py with AutoGluon Models (LOW RISK)
**UUID:** `874pSBiQzYseeW7gFawHBj`

**File:** `src/engines/engine_registry.py`

**Changes:**
```python
MODEL_FAMILIES = {
    # Existing models...

    # ADD AutoGluon models:
    "autogluon_tabular": "src.training.autogluon.tabular_trainer",
    "autogluon_timeseries": "src.training.autogluon.timeseries_trainer",
    "autogluon_bucket_specialist": "src.training.autogluon.bucket_specialist",
}
```

**Validation:**
```bash
python -c 'from src.engines.engine_registry import MODEL_FAMILIES; print(MODEL_FAMILIES)'
# Expected: AutoGluon models in registry
```

---

### Task 2.7: Create Feature Drift Detection Module (MEDIUM RISK)
**UUID:** `h6eSfMSeJGLUzhWRmqqEYq`

**File:** `src/validation/feature_drift.py`

**Purpose:** Detect feature distribution drift before training

**Implementation:**
```python
import pandas as pd
import numpy as np
from scipy.stats import entropy

def check_feature_drift(
    train_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    threshold: float = 0.1
) -> dict:
    """
    Calculate KL divergence between current and reference distributions.

    Args:
        train_df: Current training data
        reference_df: Reference training data
        threshold: KL divergence threshold (default 0.1)

    Returns:
        Dict of feature names to KL divergence values
    """
    drift_metrics = {}

    for col in train_df.columns:
        if train_df[col].dtype in ['float64', 'int64']:
            # Calculate KL divergence
            train_hist, bins = np.histogram(train_df[col], bins=50, density=True)
            ref_hist, _ = np.histogram(reference_df[col], bins=bins, density=True)

            # Add small epsilon to avoid log(0)
            train_hist += 1e-10
            ref_hist += 1e-10

            kl_div = entropy(train_hist, ref_hist)
            drift_metrics[col] = kl_div

            if kl_div > threshold:
                print(f"⚠️  Feature drift detected: {col} (KL={kl_div:.4f})")

    return drift_metrics
```

**Validation:**
```bash
python -c 'from src.validation.feature_drift import check_feature_drift; print("Drift detector ready")'
```

**Reference:** https://docs.greatexpectations.io/docs/

---

### Task 2.8: Document AutoGluon Foundation Models Setup (MEDIUM RISK)
**UUID:** `wNMujpKwWzWfdQGyG8FtBZ`

**File:** `docs/architecture/AUTOGLUON_FOUNDATION_MODELS.md`

**Sections:**
1. Overview: Foundation models vs traditional ML models
2. TabPFNv2: Installation, usage, limitations (max 10K samples)
3. Mitra: Installation, usage, CPU compatibility
4. TabICL: Installation, usage, in-context learning
5. TabM: Installation, usage, tabular transformer
6. Chronos-Bolt: Installation, usage, zero-shot time series
7. Mac M4 Compatibility: CPU-only mode, libomp requirements
8. Integration with AutoGluon: How to enable in TabularPredictor/TimeSeriesPredictor

**Validation:**
```bash
cat docs/architecture/AUTOGLUON_FOUNDATION_MODELS.md
# Expected: Documentation exists with all 6 models covered
```

---

### Task 2.9: Validate Phase 2 Complete (HIGH RISK)
**UUID:** `uoNMEUpzdBur3bPNjZXBvj`

**Validation Commands:**
```bash
# 1. Install AutoGluon
bash scripts/setup/install_autogluon_mac.sh

# 2. List available models
python -c 'from autogluon.tabular import TabularPredictor; print(TabularPredictor.list_models())'

# 3. Sync data to local DuckDB
python scripts/sync_motherduck_to_local.py

# 4. Test training (simple model, 60 second limit)
python -c '
from src.training.autogluon.tabular_trainer import train_tabular
import duckdb

conn = duckdb.connect("data/duckdb/cbi_v15.duckdb")
df = conn.execute("SELECT * FROM features.daily_ml_matrix LIMIT 1000").df()
train_tabular(df, df, "target_1w", preset="medium", time_limit=60)
'
```

**Expected Outputs:**
- ✅ AutoGluon 1.4 installed successfully
- ✅ Foundation models available (Mitra, TabPFNv2, TabICL, Chronos)
- ✅ Local DuckDB mirror synced
- ✅ Simple model trains in <60 seconds
- ✅ Model artifacts saved to data/models/

**Success Criteria:**
- No import errors
- No libomp errors on Mac M4
- Training completes successfully
- Quantile predictions (P10/P50/P90) generated

**⚠️ STOP:** Phase 3-5 depend on AutoGluon working correctly

---

## 🎯 PHASE 3: BUCKET SPECIALIST INFRASTRUCTURE

**Goal:** Build 8 bucket specialist trainers with feature selection configs
**Status:** NOT STARTED
**Dependencies:** Phase 0-2 complete

### Task 3.1: Create bucket_feature_selectors.yaml Config (LOW RISK)
**UUID:** `owV5tSmeSq96tVasgFx4hs`

**File:** `config/bucket_feature_selectors.yaml`

**Purpose:** Define feature lists for each of the Big 8 buckets

**Structure:**
```yaml
crush:
  - databento_zl_close
  - databento_zs_close
  - databento_zm_close
  - board_crush
  - oil_share
  - hog_spread
  - farmdoc_grain_outlook_sentiment

china:
  - usda_export_soybeans_weekly
  - databento_hg_close
  - china_pulse_hg_zs_correlation
  - farm_policy_news_trade_sentiment

fx:
  - fred_DX
  - databento_6l_close
  - fx_momentum_21d
  - fx_momentum_63d
  - fx_momentum_252d
  - zl_brl_correlation
  - zl_dxy_correlation

fed:
  - fred_DFEDTARU
  - fred_DGS10
  - fred_T10Y2Y
  - fred_NFCI
  - fred_STLFSI4
  - farm_policy_news_budget_sentiment

tariff:
  - scrc_tariffs_trade_sentiment
  - farm_policy_news_trade_sentiment
  - trump_truth_social_mentions

biofuel:
  - eia_rin_d3_price
  - eia_rin_d4_price
  - eia_rin_d5_price
  - eia_rin_d6_price
  - boho_spread
  - eia_biodiesel_prod
  - farmdoc_daily_rins_sentiment

energy:
  - databento_cl_close
  - databento_ho_close
  - databento_rb_close
  - crack_spread_3_2_1
  - cl_zl_correlation

volatility:
  - fred_VIXCLS
  - volatility_zl_21d
  - fred_STLFSI4
  - correlation_regime
```

**Validation:**
```bash
python -c 'import yaml; config = yaml.safe_load(open("config/bucket_feature_selectors.yaml")); print(len(config))'
# Expected: 8 buckets defined
```

---


