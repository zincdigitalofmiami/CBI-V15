# 🧠 Meta-Learning Framework - AutoML for Each Bucket (Historical Concept)

> **Important:** This document captures a **pre-V15.1** idea for building a custom AutoML system (CatBoost/TFT/Prophet/etc.) for each bucket.  
> The **actual V15.1 implementation** uses:
>
> - AutoGluon `TabularPredictor` for all Big 8 bucket specialists
> - AutoGluon `TimeSeriesPredictor` for core ZL
> - AutoGluon stacking + `WeightedEnsemble_L2` as the meta-model
>
> Do **not** create new `src/training/automl/`, `src/ensemble/`, or bespoke tournament frameworks based on this file. Treat the model/ensemble lists here as conceptual background only; implementation must follow `docs/architecture/MASTER_PLAN.md` and `AGENTS.md`.

## 🎯 The Problem

**Current approach:** "I hope PyTorch TFT works, if not we'll try again"  
**Better approach:** **Automated model selection per bucket with continuous evaluation**

---

## 🏗️ **3-Level Architecture with AutoML**

### **Level 0: Meta-Learning Layer (Model Selection)**

**For each bucket, automatically test multiple model families:**

#### **Model Candidates (12 families):**

1. **Tree-Based:**
   - CatBoost (quantile regression)
   - XGBoost (quantile regression)
   - LightGBM (quantile regression)
   - Random Forest (quantile regression via scikit-learn)

2. **Deep Learning:**
   - PyTorch TFT (Temporal Fusion Transformer)
   - PyTorch LSTM (with attention)
   - PyTorch GRU (with attention)
   - PyTorch Transformer (vanilla)

3. **Statistical:**
   - ARIMA/SARIMAX (statsmodels)
   - Prophet (Facebook)
   - GARCH (volatility modeling)

4. **Ensemble:**
   - Stacking (combine top 3 from above)

**Selection criteria:**

- Validation RMSE (P50 forecast)
- Pinball loss (P10/P90 calibration)
- Training time (< 30 min on Mac M4)
- Inference time (< 1 sec)
- Interpretability score (attention weights, SHAP values)

---

### **Level 1: Bucket-Specific Specialists (8 buckets)**

**Each bucket gets its own AutoML tournament:**

#### **Bucket 1: Crush**

**Characteristics:** Fundamental, mean-reverting, low noise  
**Best model candidates:** CatBoost, XGBoost, ARIMA  
**AutoML winner:** TBD (run tournament)

#### **Bucket 2: China**

**Characteristics:** Event-driven, regime-dependent, high noise  
**Best model candidates:** TFT, LSTM, Stacking  
**AutoML winner:** TBD (run tournament)

#### **Bucket 3: FX**

**Characteristics:** Macro-driven, trending, medium noise  
**Best model candidates:** TFT, CatBoost, Prophet  
**AutoML winner:** TBD (run tournament)

#### **Bucket 4: Fed**

**Characteristics:** Regime-dependent, low-frequency, low noise  
**Best model candidates:** CatBoost, ARIMA, Prophet  
**AutoML winner:** TBD (run tournament)

#### **Bucket 5: Tariff**

**Characteristics:** Event-driven, non-linear, Trump-era specific  
**Best model candidates:** TFT, LSTM, Stacking  
**AutoML winner:** TBD (run tournament)

#### **Bucket 6: Biofuel**

**Characteristics:** Policy-driven, seasonal, medium noise  
**Best model candidates:** TFT, CatBoost, Prophet  
**AutoML winner:** TBD (run tournament)

#### **Bucket 7: Energy**

**Characteristics:** Mean-reverting, high volatility, medium noise  
**Best model candidates:** GARCH, CatBoost, TFT  
**AutoML winner:** TBD (run tournament)

#### **Bucket 8: Volatility**

**Characteristics:** Regime-switching, non-linear, high noise (VIX, realized vol, stress indices)  
**Best model candidates:** GARCH, TFT, Stacking  
**Note:** Volatility ≠ Volume. This bucket models price variance regimes, not trading activity.  
**AutoML winner:** TBD (run tournament)

---

### **Level 2: Ensemble Aggregation (Meta-Model)**

**Instead of just Monte Carlo, test multiple ensemble methods:**

#### **Ensemble Candidates (6 methods):**

1. **Monte Carlo Simulation**
   - Sample from each bucket's P10/P50/P90
   - Run 1000 paths
   - Weight by validation performance

2. **Bayesian Model Averaging (BMA)**
   - Weight models by posterior probability
   - Accounts for model uncertainty
   - More principled than simple averaging

3. **Stacking (Meta-Learner)**
   - Train a meta-model on bucket predictions
   - Learns optimal weights automatically
   - Can be linear or non-linear (e.g., Ridge, XGBoost)

4. **Quantile Regression Averaging (QRA)**
   - Directly combine quantile forecasts
   - Preserves probabilistic calibration
   - Better than averaging point forecasts

5. **Dynamic Model Averaging (DMA)**
   - Weights change over time
   - Adapts to regime shifts
   - Uses Kalman filter or exponential smoothing

6. **Conformal Prediction**
   - Non-parametric uncertainty quantification
   - Distribution-free
   - Guaranteed coverage (e.g., 90% intervals contain true value 90% of time)

**Selection criteria:**

- Validation pinball loss (P10/P50/P90)
- Coverage (do 90% intervals actually contain 90% of outcomes?)
- Sharpness (narrower intervals = better)
- Computational cost

---

## 🤖 **AutoML Tournament Framework**

### **Phase 1: Model Selection (Per Bucket)**

```python
# Pseudocode
for bucket in ['crush', 'china', 'fx', 'fed', 'tariff', 'biofuel', 'energy', 'vol']:

    # Define model candidates
    models = [
        CatBoostQuantile(),
        XGBoostQuantile(),
        LightGBMQuantile(),
        PyTorchTFT(),
        PyTorchLSTM(),
        ARIMA(),
        Prophet(),
        GARCH(),
        StackingEnsemble()
    ]

    # Train/val split (regime-aware)
    train_data = get_train_data(bucket, regime='all')
    val_data = get_val_data(bucket, regime='current')

    # Run tournament
    results = []
    for model in models:
        # Train
        model.fit(train_data)

        # Validate
        preds = model.predict(val_data)

        # Score
        rmse = calc_rmse(preds['p50'], val_data['target'])
        pinball = calc_pinball_loss(preds, val_data['target'])
        coverage = calc_coverage(preds, val_data['target'])
        train_time = model.train_time
        inference_time = model.inference_time

        results.append({
            'model': model.name,
            'rmse': rmse,
            'pinball': pinball,
            'coverage': coverage,
            'train_time': train_time,
            'inference_time': inference_time,
            'score': weighted_score(rmse, pinball, coverage, train_time)
        })

    # Select winner
    winner = min(results, key=lambda x: x['score'])
    save_winner(bucket, winner)
```

---

### **Phase 2: Ensemble Selection**

```python
# Get predictions from all bucket winners
bucket_preds = {}
for bucket in buckets:
    model = load_winner(bucket)
    bucket_preds[bucket] = model.predict(val_data)

# Test ensemble methods
ensemble_methods = [
    MonteCarlo(),
    BayesianModelAveraging(),
    Stacking(),
    QuantileRegressionAveraging(),
    DynamicModelAveraging(),
    ConformalPrediction()
]

results = []
for method in ensemble_methods:
    # Combine bucket predictions
    final_preds = method.combine(bucket_preds)

    # Score
    pinball = calc_pinball_loss(final_preds, val_data['target'])
    coverage = calc_coverage(final_preds, val_data['target'])
    sharpness = calc_sharpness(final_preds)

    results.append({
        'method': method.name,
        'pinball': pinball,
        'coverage': coverage,
        'sharpness': sharpness,
        'score': weighted_score(pinball, coverage, sharpness)
    })

# Select winner
winner = min(results, key=lambda x: x['score'])
save_ensemble_winner(winner)
```

---

## 📊 **Continuous Evaluation & Retraining**

### **Weekly Model Health Check:**

```python
# Every week, check if current models are still best
for bucket in buckets:
    current_model = load_winner(bucket)

    # Get recent performance
    recent_data = get_recent_data(bucket, days=7)
    recent_preds = current_model.predict(recent_data)
    recent_score = calc_score(recent_preds, recent_data['target'])

    # Compare to baseline
    if recent_score > threshold:
        # Model degraded, initiate re-tournament
        run_tournament(bucket)
```

### **Monthly Full Re-Tournament:**

```python
# Every month, re-run full AutoML tournament
# This catches regime shifts, new data patterns, etc.
for bucket in buckets:
    run_tournament(bucket)

# Re-select ensemble method
run_ensemble_tournament()
```

---

## 🎯 **Why This Is Better**

### **1. No Assumptions**

- Don't assume TFT is best for tariffs
- Don't assume CatBoost is best for crush
- Let the data decide

### **2. Adaptive**

- Models can change over time
- Ensemble method can change
- Adapts to regime shifts

### **3. Robust**

- Multiple models = less overfitting
- Ensemble reduces model risk
- Continuous evaluation catches degradation

### **4. Interpretable**

- Know WHY a model was chosen (scores)
- Know WHEN it started failing (health checks)
- Know WHAT to do (re-tournament)

### **5. Scalable**

- Add new model families easily
- Add new ensemble methods easily
- Add new buckets easily

---

## 📋 **Implementation Plan**

### **Phase 1: Build AutoML Framework (Week 1-2)**

```bash
src/training/automl/
├── model_candidates.py      # All 12 model families
├── tournament.py             # AutoML tournament logic
├── scoring.py                # RMSE, pinball, coverage, etc.
└── selection.py              # Winner selection logic
```

### **Phase 2: Run Tournaments (Week 3)**

```bash
python src/training/automl/run_tournament.py --bucket crush
python src/training/automl/run_tournament.py --bucket china
# ... for all 8 buckets
```

### **Phase 3: Build Ensemble Framework (Week 4)**

```bash
src/ensemble/
├── methods.py                # All 6 ensemble methods
├── tournament.py             # Ensemble tournament
└── selection.py              # Winner selection
```

### **Phase 4: Deploy & Monitor (Week 5+)**

```bash
python src/ops/model_health_check.py  # Weekly
python src/ops/full_retournament.py   # Monthly
```

---

## ✅ **Summary**

**Instead of:**

- "I hope PyTorch works"
- Manual model selection
- Static ensemble (Monte Carlo only)

**We get:**

- Automated model selection per bucket
- 12 model families tested
- 6 ensemble methods tested
- Continuous evaluation
- Adaptive to regime shifts

**Want me to build the AutoML framework?**
