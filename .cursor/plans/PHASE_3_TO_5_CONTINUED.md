# CBI-V15.1 Implementation Plan: Phase 3-5 (Continued)

**This document continues Phase 3 and covers Phases 4-5**

---

## 🎯 PHASE 3: BUCKET SPECIALIST INFRASTRUCTURE (Continued)

### Task 3.2: Create 8 Bucket Training Configs (LOW RISK)

**UUID:** `jpZs1HoaoJ1PhZHXx582CP`

**Directory:** `config/training/buckets/`

**Files to Create:**

- `config/training/buckets/crush.yaml`
- `config/training/buckets/china.yaml`
- `config/training/buckets/fx.yaml`
- `config/training/buckets/fed.yaml`
- `config/training/buckets/tariff.yaml`
- `config/training/buckets/biofuel.yaml`
- `config/training/buckets/energy.yaml`
- `config/training/buckets/volatility.yaml`

**Template (crush.yaml):**

```yaml
bucket_name: "crush"
preset: "best_quality" # Mac M4 has no GPU
time_limit: 3600 # 1 hour
quantiles: [0.1, 0.5, 0.9]
num_stack_levels: 1 # AutoGluon creates L1 + WeightedEnsemble_L2
num_bag_folds: 8
feature_selector: "config/bucket_feature_selectors.yaml"
horizons: ["1w", "1m", "3m", "6m"]
```

**Validation:**

```bash
ls -la config/training/buckets/
# Expected: 8 YAML files created
```

---

### Task 3.3: Create bucket_specialist.py Trainer (MEDIUM RISK)

**UUID:** `v5gfQFgcjKQAexKZKYNGNy`

**File:** `src/training/autogluon/bucket_specialist.py`

**Implementation:**

```python
import yaml
from pathlib import Path
from src.training.autogluon.tabular_trainer import train_tabular

def train_bucket_specialist(
    bucket_name: str,
    train_df,
    val_df,
    config_path: str = "config/training/buckets"
):
    """
    Train bucket-specific specialist using TabularPredictor.

    Args:
        bucket_name: Name of bucket (crush, china, fx, etc.)
        train_df: Training data
        val_df: Validation data
        config_path: Path to bucket configs

    Returns:
        predictor: Trained AutoGluon TabularPredictor
        metrics: Validation metrics
    """
    # Load bucket config
    config_file = Path(config_path) / f"{bucket_name}.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Load feature selector
    with open(config['feature_selector']) as f:
        feature_selectors = yaml.safe_load(f)

    # Filter features to bucket-specific subset
    bucket_features = feature_selectors[bucket_name]
    train_df_filtered = train_df[bucket_features + ['target_1w']]
    val_df_filtered = val_df[bucket_features + ['target_1w']]

    # Train TabularPredictor
    predictor = train_tabular(
        train_df=train_df_filtered,
        val_df=val_df_filtered,
        target_col='target_1w',
        preset=config['preset'],
        quantiles=config['quantiles'],
        time_limit=config['time_limit'],
        model_path=f"data/models/bucket_{bucket_name}"
    )

    # Get OOF predictions
    oof_preds = predictor.predict(val_df_filtered, as_pandas=True)

    # Save to training.bucket_predictions table
    save_bucket_predictions(bucket_name, oof_preds)

    # Get feature importance
    feature_importance = predictor.feature_importance(val_df_filtered)

    return predictor, {
        'bucket_name': bucket_name,
        'validation_pinball_loss': predictor.evaluate(val_df_filtered),
        'feature_importance': feature_importance
    }
```

**Validation:**

```bash
python -c 'from src.training.autogluon.bucket_specialist import train_bucket_specialist; print("Trainer ready")'
```

---

### Task 3.4: Create train_all_buckets.py Orchestrator (HIGH RISK)

**UUID:** `jDE5TBnCkruM53X73oScpE`

**File:** `src/training/autogluon/train_all_buckets.py`

**Purpose:** Orchestrate training of all 8 bucket specialists + main predictor + Chronos baseline

**Implementation:**

```python
import argparse
from multiprocessing import Pool
from src.training.autogluon.bucket_specialist import train_bucket_specialist
from src.training.autogluon.tabular_trainer import train_tabular
from src.training.autogluon.timeseries_trainer import train_timeseries
import duckdb

BUCKETS = ['crush', 'china', 'fx', 'fed', 'tariff', 'biofuel', 'energy', 'volatility']

def train_single_bucket(bucket_name):
    """Train a single bucket specialist."""
    conn = duckdb.connect('data/duckdb/cbi_v15.duckdb')
    train_df = conn.execute("SELECT * FROM features.daily_ml_matrix WHERE split = 'train'").df()
    val_df = conn.execute("SELECT * FROM features.daily_ml_matrix WHERE split = 'val'").df()
    conn.close()

    return train_bucket_specialist(bucket_name, train_df, val_df)

def main(args):
    # 1. Sync MotherDuck → Local DuckDB
    print("Syncing MotherDuck → Local DuckDB...")
    import subprocess
    subprocess.run(['python', 'scripts/sync_motherduck_to_local.py'], check=True)

    # 2. Train 8 bucket specialists (parallel if requested)
    print(f"Training {len(BUCKETS)} bucket specialists...")
    if args.parallel:
        with Pool(processes=4) as pool:
            results = pool.map(train_single_bucket, BUCKETS)
    else:
        results = [train_single_bucket(b) for b in BUCKETS]

    # 3. Train main ZL predictor (all features)
    print("Training main ZL predictor...")
    conn = duckdb.connect('data/duckdb/cbi_v15.duckdb')
    train_df = conn.execute("SELECT * FROM features.daily_ml_matrix WHERE split = 'train'").df()
    val_df = conn.execute("SELECT * FROM features.daily_ml_matrix WHERE split = 'val'").df()
    conn.close()

    main_predictor = train_tabular(
        train_df, val_df, 'target_1w',
        preset='best_quality',
        time_limit=args.time_limit
    )

    # 4. Train Chronos-Bolt baseline (zero-shot)
    print("Training Chronos-Bolt baseline...")
    chronos_predictor = train_timeseries(
        train_df, 'target_1w',
        known_covariates=[f'{b}_bucket_score' for b in BUCKETS],
        prediction_length=5
    )

    # 5. Upload results to MotherDuck
    print("Uploading results to MotherDuck...")
    upload_to_motherduck()

    print(f"✅ Training complete: {len(BUCKETS)} buckets + main + Chronos")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--time-limit', type=int, default=7200)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        print(f"Dry run: Would train {len(BUCKETS)} buckets + main + Chronos")
    else:
        main(args)
```

**Validation:**

```bash
python src/training/autogluon/train_all_buckets.py --dry-run
# Expected: Dry run shows 10 models to train (8 buckets + main + Chronos)
```

**Risk:** HIGH - Parallel training may exhaust Mac M4 resources

---

### Task 3.5: Validate Phase 3 Complete (CRITICAL)

**UUID:** `g75aHVzqaYgdWvYS6b7mkV`

**Validation Commands:**

```bash
# 1. Sync data
python scripts/sync_motherduck_to_local.py

# 2. Train all buckets
python src/training/autogluon/train_all_buckets.py --time-limit 3600

# 3. Check bucket predictions
SELECT COUNT(DISTINCT bucket_name) FROM training.bucket_predictions;
# Expected: 8

# 4. Check predictions per bucket
SELECT bucket_name, COUNT(*) as predictions
FROM training.bucket_predictions
GROUP BY bucket_name;

# 5. Check model artifacts
ls -la data/models/bucket_*/
```

**Expected Outputs:**

- ✅ 8 bucket specialists trained successfully
- ✅ 1 main ZL predictor trained
- ✅ 1 Chronos-Bolt baseline trained
- ✅ OOF predictions in training.bucket_predictions (8 buckets)
- ✅ Model artifacts in data/models/bucket\_{name}/
- ✅ Feature importance available for each bucket

**Success Criteria:**

- All 8 buckets have trained models
- OOF predictions cover full training period
- Validation pinball loss < baseline
- No memory errors on Mac M4

**Big 8 Impact:** All 8 buckets validated

**⚠️ STOP:** Phase 4-5 depend on bucket specialists working correctly

---

## 🔗 PHASE 4: AUTOGLUON STACKING & MONTE CARLO

**Goal:** Build L1 stacking layer, L2.5 greedy ensemble, and L3 Monte Carlo simulation
**Status:** NOT STARTED
**Dependencies:** Phase 0-3 complete

### Task 4.1: Create L1 Stacking Layer (MEDIUM RISK)

**UUID:** `kPK2BQMnyUnBFejmNJVa8E` (reused UUID from TSci removal task)

**Purpose:** AutoGluon automatically creates L1 stacking when `num_stack_levels=1`

**File:** `src/training/autogluon/stacking_layer.py`

**Implementation:**

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

def create_stacking_layer(
    bucket_predictions: pd.DataFrame,
    main_predictions: pd.DataFrame,
    chronos_predictions: pd.DataFrame,
    target_col: str = 'target_1w'
):
    """
    Create L1 stacking layer from L0 OOF predictions.

    L0 inputs (10 models):
    - 8 bucket specialists (crush, china, fx, fed, tariff, biofuel, energy, volatility)
    - 1 main ZL predictor (all features)
    - 1 Chronos-Bolt baseline (zero-shot)

    L1 output:
    - AutoGluon stacking layer (trains on L0 OOF predictions)
    - Automatically creates WeightedEnsemble_L2

    Args:
        bucket_predictions: OOF predictions from 8 bucket specialists
        main_predictions: OOF predictions from main ZL predictor
        chronos_predictions: Predictions from Chronos-Bolt
        target_col: Target column name

    Returns:
        l1_predictor: Trained L1 stacking layer
    """
    # Combine L0 predictions into single DataFrame
    l0_features = pd.concat([
        bucket_predictions.pivot(columns='bucket_name', values='prediction'),
        main_predictions[['prediction']].rename(columns={'prediction': 'main_pred'}),
        chronos_predictions[['prediction']].rename(columns={'prediction': 'chronos_pred'})
    ], axis=1)

    # Add target
    l0_features[target_col] = bucket_predictions[target_col]

    # Train L1 stacking layer
    l1_predictor = TabularPredictor(
        label=target_col,
        problem_type='quantile',
        quantile_levels=[0.1, 0.5, 0.9],
        path='data/models/l1_stacking'
    )

    l1_predictor.fit(
        train_data=l0_features,
        presets='best_quality',
        time_limit=1800,
        num_stack_levels=1  # Creates WeightedEnsemble_L2 automatically
    )

    return l1_predictor
```

**Validation:**

```bash
python -c 'from src.training.autogluon.stacking_layer import create_stacking_layer; print("L1 stacking ready")'
```

---

### Task 4.2: Create L2.5 Greedy Weighted Ensemble (CRITICAL - UPGRADED FEATURE)

**UUID:** `qGNNkACKGH6SYY6swMcHXG` (reused UUID)

**Purpose:** User explicitly wants greedy ensemble as "UPGRADED FEATURE" beyond AutoGluon's default

**File:** `src/training/autogluon/greedy_ensemble.py`

**Implementation:**

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def greedy_weighted_ensemble(
    l2_predictions: pd.DataFrame,
    chronos_predictions: pd.DataFrame,
    bucket_predictions: pd.DataFrame,
    target: pd.Series,
    quantiles: list = [0.1, 0.5, 0.9]
):
    """
    L2.5 Greedy Weighted Ensemble (UPGRADED FEATURE).

    Combines:
    - L2 WeightedEnsemble_L2 (AutoGluon's automatic ensemble)
    - Chronos-Bolt zero-shot baseline
    - 8 bucket specialists (for diversity)

    Optimization:
    - Minimize pinball loss across all quantiles
    - Greedy search for optimal weights
    - Constraints: weights sum to 1, all weights >= 0

    Args:
        l2_predictions: Predictions from AutoGluon WeightedEnsemble_L2
        chronos_predictions: Predictions from Chronos-Bolt
        bucket_predictions: Predictions from 8 bucket specialists
        target: True target values
        quantiles: Quantile levels

    Returns:
        weights: Optimal weights for each model
        ensemble_predictions: Final ensemble predictions
    """
    # Stack all predictions
    all_preds = np.column_stack([
        l2_predictions.values,
        chronos_predictions.values,
        bucket_predictions.values
    ])

    # Pinball loss function
    def pinball_loss(weights, preds, target, quantile):
        ensemble_pred = preds @ weights
        error = target - ensemble_pred
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

    # Optimize weights for each quantile
    optimal_weights = {}
    for q in quantiles:
        result = minimize(
            fun=lambda w: pinball_loss(w, all_preds, target, q),
            x0=np.ones(all_preds.shape[1]) / all_preds.shape[1],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            bounds=[(0, 1)] * all_preds.shape[1],
            method='SLSQP'
        )
        optimal_weights[f'q{int(q*100)}'] = result.x

    # Generate ensemble predictions
    ensemble_predictions = {}
    for q, weights in optimal_weights.items():
        ensemble_predictions[q] = all_preds @ weights

    return optimal_weights, pd.DataFrame(ensemble_predictions)
```

**Validation:**

```bash
python -c 'from src.training.autogluon.greedy_ensemble import greedy_weighted_ensemble; print("Greedy ensemble ready")'
```

**Reference:** https://arxiv.org/abs/2003.03186 (Greedy Ensemble Learning)

---

### Task 4.3: Create L3 Monte Carlo Simulation (CRITICAL)

**UUID:** `237KEwaYT5cJ4J2vBQWjoH` (reused UUID)

**Purpose:** Generate probabilistic scenarios for VaR/CVaR risk metrics

**File:** `src/training/autogluon/monte_carlo.py`

**Implementation:**

```python
import numpy as np
import pandas as pd
from scipy.stats import norm

def monte_carlo_simulation(
    ensemble_predictions: pd.DataFrame,
    n_simulations: int = 10000,
    confidence_levels: list = [0.95, 0.99]
):
    """
    L3 Monte Carlo Simulation for probabilistic scenarios.

    Inputs:
    - L2.5 greedy ensemble predictions (P10, P50, P90)

    Outputs:
    - VaR (Value at Risk) at 95% and 99% confidence
    - CVaR (Conditional Value at Risk) at 95% and 99%
    - Full distribution of simulated outcomes

    Args:
        ensemble_predictions: DataFrame with q10, q50, q90 columns
        n_simulations: Number of Monte Carlo simulations
        confidence_levels: Confidence levels for VaR/CVaR

    Returns:
        scenarios: DataFrame with simulated scenarios
        risk_metrics: Dict with VaR/CVaR values
    """
    # Extract quantiles
    q10 = ensemble_predictions['q10'].values
    q50 = ensemble_predictions['q50'].values
    q90 = ensemble_predictions['q90'].values

    # Estimate distribution parameters (assume normal)
    # P10 = μ - 1.28σ, P50 = μ, P90 = μ + 1.28σ
    mu = q50
    sigma = (q90 - q10) / (2 * 1.28)

    # Generate simulations
    scenarios = np.random.normal(
        loc=mu[:, np.newaxis],
        scale=sigma[:, np.newaxis],
        size=(len(mu), n_simulations)
    )

    # Calculate VaR and CVaR
    risk_metrics = {}
    for conf in confidence_levels:
        var_threshold = np.percentile(scenarios, (1 - conf) * 100, axis=1)
        cvar = np.mean(scenarios[scenarios <= var_threshold[:, np.newaxis]], axis=1)

        risk_metrics[f'VaR_{int(conf*100)}'] = var_threshold
        risk_metrics[f'CVaR_{int(conf*100)}'] = cvar

    return pd.DataFrame(scenarios), risk_metrics
```

**Validation:**

```bash
python -c 'from src.training.autogluon.monte_carlo import monte_carlo_simulation; print("Monte Carlo ready")'
```

**Reference:** https://en.wikipedia.org/wiki/Value_at_risk

---

### Task 4.4: Create Full Pipeline Orchestrator (HIGH RISK)

**UUID:** `wL9sbRjMLzcGRni2RHnirw` (reused UUID)

**File:** `src/training/autogluon/full_pipeline.py`

**Purpose:** Orchestrate L0 → L1 → L2 → L2.5 → L3 pipeline

**Implementation:**

```python
from src.training.autogluon.train_all_buckets import train_single_bucket
from src.training.autogluon.stacking_layer import create_stacking_layer
from src.training.autogluon.greedy_ensemble import greedy_weighted_ensemble
from src.training.autogluon.monte_carlo import monte_carlo_simulation

def run_full_pipeline(dry_run=False):
    """
    Run full AutoGluon hybrid pipeline:

    L0: 8 bucket specialists + 1 main predictor + 1 Chronos baseline = 10 models
    L1: AutoGluon stacking layer (trains on L0 OOF predictions)
    L2: WeightedEnsemble_L2 (automatically created by AutoGluon)
    L2.5: Greedy weighted ensemble (UPGRADED FEATURE)
    L3: Monte Carlo simulation (VaR/CVaR)

    Args:
        dry_run: If True, only print pipeline steps without training

    Returns:
        final_predictions: DataFrame with P10/P50/P90 + VaR/CVaR
    """
    if dry_run:
        print("DRY RUN: Full pipeline steps:")
        print("  L0: Train 8 buckets + main + Chronos (10 models)")
        print("  L1: Train stacking layer on L0 OOF predictions")
        print("  L2: AutoGluon creates WeightedEnsemble_L2")
        print("  L2.5: Greedy weighted ensemble optimization")
        print("  L3: Monte Carlo simulation (10K scenarios)")
        return

    # L0: Train all base models
    print("L0: Training 8 buckets + main + Chronos...")
    # (use train_all_buckets.py)

    # L1: Create stacking layer
    print("L1: Creating stacking layer...")
    l1_predictor = create_stacking_layer(...)

    # L2: AutoGluon automatically creates WeightedEnsemble_L2
    print("L2: WeightedEnsemble_L2 created automatically")

    # L2.5: Greedy ensemble
    print("L2.5: Optimizing greedy weighted ensemble...")
    weights, ensemble_preds = greedy_weighted_ensemble(...)

    # L3: Monte Carlo
    print("L3: Running Monte Carlo simulation (10K scenarios)...")
    scenarios, risk_metrics = monte_carlo_simulation(ensemble_preds)

    # Combine final predictions
    final_predictions = pd.concat([
        ensemble_preds,
        pd.DataFrame(risk_metrics)
    ], axis=1)

    # Upload to MotherDuck
    upload_to_motherduck(final_predictions, 'forecasts.zl_predictions')

    return final_predictions
```

**Validation:**

```bash
python -c 'from src.training.autogluon.full_pipeline import run_full_pipeline; run_full_pipeline(dry_run=True)'
# Expected: Prints 5-layer pipeline steps
```

---

### Task 4.5: Validate Phase 4 Complete (CRITICAL)

**UUID:** `t9nbeyciHZfQHJgBP7qoWM` (reused UUID)

**Validation Commands:**

```bash
# 1. Run full pipeline
python src/training/autogluon/full_pipeline.py

# 2. Check final predictions
SELECT COUNT(*) FROM forecasts.zl_predictions;
# Expected: Predictions for all validation dates

# 3. Check quantile coverage
SELECT
  AVG(CASE WHEN actual < q10 THEN 1 ELSE 0 END) as below_q10,
  AVG(CASE WHEN actual > q90 THEN 1 ELSE 0 END) as above_q90
FROM forecasts.zl_predictions;
# Expected: ~10% below Q10, ~10% above Q90

# 4. Check VaR/CVaR
SELECT AVG(VaR_95), AVG(CVaR_95) FROM forecasts.zl_predictions;
# Expected: Reasonable risk metrics
```

**Expected Outputs:**

- ✅ L0: 10 models trained (8 buckets + main + Chronos)
- ✅ L1: Stacking layer trained on L0 OOF predictions
- ✅ L2: WeightedEnsemble_L2 created automatically
- ✅ L2.5: Greedy ensemble weights optimized
- ✅ L3: Monte Carlo scenarios generated (10K simulations)
- ✅ Final predictions in forecasts.zl_predictions

**Success Criteria:**

- Quantile calibration: ~10% below Q10, ~10% above Q90
- Pinball loss < baseline
- VaR/CVaR metrics reasonable
- No memory errors on Mac M4

**⚠️ STOP:** Phase 5 depends on full pipeline working correctly

---

## 🚀 PHASE 5: TRIGGER.DEV ORCHESTRATION

**Goal:** Automate daily training, forecasting, and monitoring with Trigger.dev
**Status:** NOT STARTED
**Dependencies:** Phase 0-4 complete

### Task 5.1: Create Daily Training Trigger Job (HIGH RISK)

**UUID:** `nmDVCuPjvqA2A9XRA6YD7J` (reused UUID)

**File:** `trigger/daily_training.ts`

**Purpose:** Automated daily training of full pipeline

**Schedule:** Daily at 2 AM UTC (after all data feeds updated)

**Implementation:**

```typescript
import { task, schedules } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

export const dailyTraining = task({
  id: "daily-training",
  run: async (payload, { ctx }) => {
    try {
      // 1. Sync MotherDuck → Local DuckDB
      console.log("Syncing MotherDuck → Local DuckDB...");
      await execAsync("python scripts/sync_motherduck_to_local.py");

      // 2. Run full pipeline
      console.log("Running full AutoGluon pipeline...");
      const { stdout, stderr } = await execAsync("python src/training/autogluon/full_pipeline.py");

      // 3. Upload results to MotherDuck
      console.log("Uploading results to MotherDuck...");
      await execAsync("python scripts/upload_to_motherduck.py");

      // 4. Send success notification
      await sendSuccessNotification({
        job: "daily-training",
        timestamp: new Date(),
        models_trained: 10,
        predictions_generated: true,
      });

      return { success: true, stdout, stderr };
    } catch (error) {
      // Send failure alert
      await sendFailureAlert({
        job: "daily-training",
        error: error.message,
        stack: error.stack,
        timestamp: new Date(),
        severity: "CRITICAL",
      });
      throw error;
    }
  },
});

// Schedule: Daily at 2 AM UTC
schedules.create({
  id: "daily-training-schedule",
  cron: "0 2 * * *",
  task: dailyTraining,
});
```

**Validation:**

```bash
# Test trigger job locally
npx trigger.dev@latest dev
# Then trigger manually in Trigger.dev dashboard
```

---

### Task 5.2: Create Daily Forecast Generation Job (MEDIUM RISK)

**UUID:** `vuAnAqaTPqVFmUr48UQ1xr` (reused UUID)

**File:** `trigger/daily_forecast.ts`

**Purpose:** Generate daily forecasts using trained models

**Schedule:** Daily at 10 AM UTC (after market open)

**Implementation:**

```typescript
export const dailyForecast = task({
  id: "daily-forecast",
  run: async () => {
    // 1. Get latest data
    const latestData = await getLatestFeatures();

    // 2. Load trained models
    const models = await loadTrainedModels();

    // 3. Generate predictions
    const predictions = await generatePredictions(models, latestData);

    // 4. Run Monte Carlo simulation
    const scenarios = await runMonteCarloSimulation(predictions);

    // 5. Upload to forecasts.zl_predictions
    await uploadForecasts(predictions, scenarios);

    // 6. Trigger dashboard refresh
    await triggerDashboardRefresh();

    return { forecastsGenerated: predictions.length };
  },
});

schedules.create({
  id: "daily-forecast-schedule",
  cron: "0 10 * * *",
  task: dailyForecast,
});
```

---

### Task 5.3: Create Model Performance Monitoring Job (MEDIUM RISK)

**UUID:** `efED7rpvbH4E4ejX9KVdcS` (reused UUID)

**File:** `trigger/model_monitoring.ts`

**Purpose:** Monitor model performance and detect drift

**Schedule:** Daily at 6 PM UTC (after market close)

**Implementation:**

```typescript
export const modelMonitoring = task({
  id: "model-monitoring",
  run: async () => {
    // 1. Calculate prediction errors
    const errors = await calculatePredictionErrors();

    // 2. Check feature drift
    const drift = await checkFeatureDrift();

    // 3. Check quantile calibration
    const calibration = await checkQuantileCalibration();

    // 4. Alert if performance degraded
    if (errors.mae > THRESHOLD || drift.max_kl > 0.1) {
      await sendPerformanceAlert({
        mae: errors.mae,
        max_drift: drift.max_kl,
        calibration: calibration,
        severity: "WARNING",
      });
    }

    // 5. Log metrics to MotherDuck
    await logMetrics(errors, drift, calibration);

    return { status: "monitored", errors, drift, calibration };
  },
});

schedules.create({
  id: "model-monitoring-schedule",
  cron: "0 18 * * *",
  task: modelMonitoring,
});
```

---

### Task 5.4: Create Weekly Retraining Job (LOW RISK)

**UUID:** `chRBL6LYNgCEac3Zu5D8Gc` (reused UUID)

**File:** `trigger/weekly_retraining.ts`

**Purpose:** Full retraining of all models weekly

**Schedule:** Sunday at 12 AM UTC

**Implementation:**

```typescript
export const weeklyRetraining = task({
  id: "weekly-retraining",
  run: async () => {
    // 1. Sync all data from MotherDuck
    await execAsync("python scripts/sync_motherduck_to_local.py");

    // 2. Run full pipeline with extended time limits
    await execAsync("python src/training/autogluon/full_pipeline.py --time-limit 14400");

    // 3. Validate new models
    const validation = await validateNewModels();

    // 4. If validation passes, deploy new models
    if (validation.passed) {
      await deployNewModels();
      await sendDeploymentNotification();
    } else {
      await sendValidationFailureAlert(validation);
    }

    return { retrainingComplete: true, validation };
  },
});

schedules.create({
  id: "weekly-retraining-schedule",
  cron: "0 0 * * 0", // Sunday midnight
  task: weeklyRetraining,
});
```

---

### Task 5.5: Create Data Quality Monitoring Job (HIGH RISK)

**UUID:** `874pSBiQzYseeW7gFawHBj` (reused UUID)

**File:** `trigger/data_quality_monitoring.ts`

**Purpose:** Monitor data quality across all sources

**Schedule:** Hourly

**Implementation:**

```typescript
export const dataQualityMonitoring = task({
  id: "data-quality-monitoring",
  run: async () => {
    const sources = [
      "databento",
      "fred",
      "eia",
      "epa",
      "usda",
      "cftc",
      "farm_policy_news",
      "farmdoc_daily",
    ];

    const results = {};

    for (const source of sources) {
      // Check data freshness
      const freshness = await checkDataFreshness(source);

      // Check data quality (6 dimensions)
      const quality = await checkDataQuality(source);

      results[source] = { freshness, quality };

      // Alert if stale or poor quality
      if (!freshness.is_fresh || !quality.all_passed) {
        await sendDataQualityAlert({
          source,
          freshness,
          quality,
          severity: "WARNING",
        });
      }
    }

    return { sources: sources.length, results };
  },
});

schedules.create({
  id: "data-quality-monitoring-schedule",
  cron: "0 * * * *", // Hourly
  task: dataQualityMonitoring,
});
```

---

### Task 5.6: Create Notification System (MEDIUM RISK)

**UUID:** `h6eSfMSeJGLUzhWRmqqEYq` (reused UUID)

**File:** `trigger/notifications.ts`

**Purpose:** Centralized notification system for all alerts

**Implementation:**

```typescript
export async function sendSuccessNotification(payload) {
  // Send to Slack, email, or other channels
  await fetch(process.env.SLACK_WEBHOOK_URL, {
    method: "POST",
    body: JSON.stringify({
      text: `✅ ${payload.job} completed successfully`,
      attachments: [
        {
          color: "good",
          fields: [
            { title: "Timestamp", value: payload.timestamp },
            { title: "Models Trained", value: payload.models_trained },
          ],
        },
      ],
    }),
  });
}

export async function sendFailureAlert(payload) {
  await fetch(process.env.SLACK_WEBHOOK_URL, {
    method: "POST",
    body: JSON.stringify({
      text: `🚨 ${payload.job} FAILED`,
      attachments: [
        {
          color: "danger",
          fields: [
            { title: "Error", value: payload.error },
            { title: "Severity", value: payload.severity },
            { title: "Timestamp", value: payload.timestamp },
          ],
        },
      ],
    }),
  });
}

export async function sendPerformanceAlert(payload) {
  // Similar implementation for performance degradation alerts
}

export async function sendDataQualityAlert(payload) {
  // Similar implementation for data quality alerts
}
```

---

### Task 5.7: Validate Phase 5 Complete (CRITICAL)

**UUID:** `wNMujpKwWzWfdQGyG8FtBZ` (reused UUID)

**Validation Commands:**

```bash
# 1. Deploy Trigger.dev jobs
npx trigger.dev@latest deploy

# 2. Test daily training job
npx trigger.dev@latest test daily-training

# 3. Test daily forecast job
npx trigger.dev@latest test daily-forecast

# 4. Test monitoring job
npx trigger.dev@latest test model-monitoring

# 5. Check Trigger.dev dashboard
# Expected: All jobs scheduled and running
```

**Expected Outputs:**

- ✅ 5 Trigger.dev jobs deployed
- ✅ Daily training runs at 2 AM UTC
- ✅ Daily forecast runs at 10 AM UTC
- ✅ Model monitoring runs at 6 PM UTC
- ✅ Weekly retraining runs Sunday midnight
- ✅ Data quality monitoring runs hourly
- ✅ Notifications sent to Slack on success/failure

**Success Criteria:**

- All jobs execute without errors
- Notifications received for all events
- Models retrained weekly
- Forecasts generated daily
- Performance monitored continuously

**🎉 PHASE 5 COMPLETE:** Full production system operational

---

## 📊 FINAL VALIDATION CHECKLIST

### System Architecture Validation

- [ ] MotherDuck (cloud) + Local DuckDB (training) working
- [ ] 8 schemas created (raw, staging, features, training, forecasts, etc.)
- [ ] 23 SQL files in database/models/ (MANIFEST.md verified)
- [ ] No references to database/definitions/ (except deprecation notice)
- [ ] No TSci references (all removed)
- [ ] No BigQuery/Dataform references (except "we don't use it")

### Data Pipeline Validation

- [ ] All 8 Big 8 buckets have required data sources
- [ ] EPA RIN prices ingested (weekly since 2010)
- [ ] USDA export sales (no mock data)
- [ ] CFTC COT coverage (38 symbols)
- [ ] FRED active series (24+ series)
- [ ] Farm Policy News + farmdoc Daily scrapers working
- [ ] Data quality checks (6 dimensions) passing

### AutoGluon Validation

- [ ] AutoGluon 1.4 installed on Mac M4 (no libomp errors)
- [ ] TabularPredictor working (quantile regression)
- [ ] TimeSeriesPredictor working (Chronos-Bolt)
- [ ] Foundation models available (Mitra, TabPFNv2, TabICL, TabM)
- [ ] 8 bucket specialists trained
- [ ] Main ZL predictor trained
- [ ] Chronos-Bolt baseline trained

### Ensemble Validation

- [ ] L0: 10 models (8 buckets + main + Chronos)
- [ ] L1: Stacking layer trained on L0 OOF predictions
- [ ] L2: WeightedEnsemble_L2 created automatically
- [ ] L2.5: Greedy weighted ensemble optimized
- [ ] L3: Monte Carlo simulation (10K scenarios)
- [ ] Quantile calibration: ~10% below Q10, ~10% above Q90
- [ ] VaR/CVaR metrics reasonable

### Trigger.dev Validation

- [ ] 5 Trigger.dev jobs deployed
- [ ] Daily training scheduled (2 AM UTC)
- [ ] Daily forecast scheduled (10 AM UTC)
- [ ] Model monitoring scheduled (6 PM UTC)
- [ ] Weekly retraining scheduled (Sunday midnight)
- [ ] Data quality monitoring scheduled (hourly)
- [ ] Notifications working (Slack/email)

### Dashboard Validation

- [ ] Next.js dashboard deployed to Vercel
- [ ] Queries forecasts.zl_predictions from MotherDuck
- [ ] Displays P10/P50/P90 forecasts
- [ ] Displays VaR/CVaR risk metrics
- [ ] Displays Big 8 bucket scores
- [ ] Displays model performance metrics

---

## 🎯 SUCCESS CRITERIA (FINAL)

**System is production-ready when:**

1. ✅ All 49 tasks across 6 phases (Phase -1 through Phase 5) complete
2. ✅ All validation checklists pass
3. ✅ Daily forecasts generated automatically
4. ✅ Models retrained weekly
5. ✅ Performance monitored continuously
6. ✅ Dashboard displays live forecasts
7. ✅ No critical bugs or errors
8. ✅ Documentation complete and up-to-date

**🚀 CBI-V15.1 PRODUCTION DEPLOYMENT COMPLETE**
