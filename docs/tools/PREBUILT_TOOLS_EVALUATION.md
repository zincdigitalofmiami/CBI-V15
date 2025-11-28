# Pre-Built Tools Evaluation - Fit Assessment

**Date**: November 28, 2025  
**Status**: ✅ **EVALUATING** - Against methodology, goals, cost constraints  
**Principle**: No bloat, no reckless overtooling

---

## 🎯 Evaluation Criteria

### Fit Assessment:
1. ✅ **Aligns with Mac Training** (not cloud-dependent)
2. ✅ **Institutional-Grade** (GS Quant/JPM standards)
3. ✅ **Cost-Effective** (fits $50/month GCP cap)
4. ✅ **Avoids Bloat** (adds value, not complexity)
5. ✅ **Fits Architecture** (BigQuery + Dataform + Mac training)

---

## ✅ APPROVED: High-Value, Low-Bloat Tools

### 1. Pandas-TA ✅ **APPROVED**

**What**: Vectorized TA library (130+ indicators) with numba/numpy

**Fit**: ✅ **PERFECT**
- Already in our plan (`requirements.txt`)
- Matches TradingView/Bloomberg definitions
- Pandas-native (fits Mac training)
- Faster than custom loops
- Good for Family A (technicals)

**Use Case**:
- Mac training: Calculate technical indicators locally
- BigQuery: Pre-compute in SQL (already planned)
- Hybrid: Use pandas-ta for validation/comparison

**Cost**: Free (open source)

**Action**: ✅ **ALREADY PLANNED** - Add to `requirements.txt`

---

### 2. Pandera ✅ **APPROVED** - CRITICAL

**What**: DataFrame validation (Pydantic for Pandas)

**Fit**: ✅ **PERFECT**
- Prevents logic inversions (China sentiment bug)
- Lightweight, no cloud dependency
- Fits Mac training pipeline
- Can hard-code economic assumptions as unit tests

**Use Case**:
- Mac training: Validate input matrix before training
- Feature engineering: Check correlations (e.g., China sentiment must be positive with ZL returns)
- Pipeline guardrails: Fail fast if logic inverted

**Code Example**:
```python
import pandera as pa

schema = pa.DataFrameSchema({
    "china_sentiment": pa.Column(
        float, 
        checks=pa.Check(
            lambda g: g.corr(df['zl_return']) > 0, 
            error="CRITICAL: China Sentiment negatively correlated. Logic inverted?"
        )
    ),
    "crush_margin": pa.Column(
        float,
        checks=pa.Check(lambda x: x > -100, error="Crush margin too negative")
    )
})
```

**Cost**: Free (open source)

**Action**: ✅ **ADD IMMEDIATELY** - Add to `requirements.txt` and Mac training scripts

---

### 3. pycot-reports ✅ **APPROVED**

**What**: Downloads, parses, and cleans CFTC COT reports

**Fit**: ✅ **PERFECT**
- Solves CFTC ingestion nightmare
- Handles Legacy vs. Disaggregated split automatically
- 3 lines of code vs. weeks of parsing
- Fits Mac ingestion scripts

**Use Case**:
- `src/ingestion/cftc/collect_cftc_comprehensive.py`
- Replace manual parsing with `pycot_reports.get_cot_report()`

**Cost**: Free (open source)

**Action**: ✅ **ADD IMMEDIATELY** - Integrate into CFTC ingestion script

---

### 4. wasdeparser ✅ **APPROVED**

**What**: Scrapes and parses USDA WASDE text/XML archives

**Fit**: ✅ **PERFECT**
- Solves USDA parsing nightmare
- Handles "Revisionist History" (pull specific historical dates)
- Fits Mac ingestion scripts

**Use Case**:
- `src/ingestion/usda/collect_usda_comprehensive.py`
- Replace manual parsing with `wasdeparser.parse_wasde()`

**Cost**: Free (open source)

**Action**: ✅ **ADD IMMEDIATELY** - Integrate into USDA ingestion script

---

### 5. SHAP ✅ **APPROVED** - CRITICAL

**What**: Explainable AI (SHAP values for model interpretability)

**Fit**: ✅ **PERFECT**
- Already in our plan (feature importance)
- Fits Mac training pipeline
- Detects logic inversions (slope analysis)
- Industry standard (GS Quant uses SHAP)

**Use Case**:
- Post-training: SHAP dependence plots for sentiment features
- Feature validation: Check if slopes match economic theory
- Model interpretability: Explain LightGBM predictions

**Cost**: Free (open source)

**Action**: ✅ **ALREADY PLANNED** - Ensure in `requirements.txt`

---

## ⚠️ EVALUATE: Medium-Value, Context-Dependent

### 6. ruptures ⚠️ **EVALUATE**

**What**: Offline Change Point Detection (finds structural breaks)

**Fit**: ⚠️ **MAYBE**
- We already have manual regime calendar
- Could auto-label regimes (reduce bias)
- But: Manual regimes are domain knowledge (Trump eras, crises)

**Use Case**:
- Research: Auto-detect volatility regime shifts
- Validation: Compare manual regimes vs. detected breaks
- Feature engineering: Add detected breakpoints as features

**Cost**: Free (open source)

**Decision**: ⚠️ **DEFER** - Use manual regimes for now, add ruptures later for validation

**Action**: ⚠️ **OPTIONAL** - Add to research scripts, not production pipeline

---

### 7. Alphalens-Reloaded ⚠️ **EVALUATE**

**What**: Factor analysis (quantile spread analysis)

**Fit**: ⚠️ **MAYBE**
- Detects logic inversions (would catch China bug)
- Post-feature engineering validation
- But: Adds complexity, might be overkill

**Use Case**:
- Feature validation: Check if sentiment features have correct quantile spreads
- Research: Factor analysis before training

**Cost**: Free (open source)

**Decision**: ⚠️ **DEFER** - Pandera + SHAP might be sufficient

**Action**: ⚠️ **OPTIONAL** - Add to research/validation scripts

---

### 8. pymicrostructure ⚠️ **EVALUATE**

**What**: VPIN, Amihud Illiquidity, Roll Models

**Fit**: ⚠️ **MAYBE**
- We already have Amihud in technical indicators
- VPIN is deferred (microstructure not in Phase 1)
- Could be useful for advanced features

**Use Case**:
- Advanced features: VPIN, Roll Models (if we add microstructure)
- Validation: Compare our Amihud vs. pymicrostructure

**Cost**: Free (open source)

**Decision**: ⚠️ **DEFER** - We already have Amihud, VPIN is deferred

**Action**: ⚠️ **OPTIONAL** - Add when we implement microstructure features

---

## ❌ REJECTED: Bloat, Cost, or Misalignment

### 9. mlfinlab ❌ **REJECTED**

**What**: Commercial library (Hudson & Thames)

**Fit**: ❌ **REJECTED**
- Commercial (costs money)
- We already have Garman-Klass, Parkinson in BigQuery SQL
- Free alternatives exist (ta-lib, tsfracdiff)

**Alternative**: Use ta-lib or tsfracdiff (free)

**Action**: ❌ **DO NOT ADD**

---

### 10. Great Expectations ❌ **REJECTED**

**What**: Data pipeline testing framework

**Fit**: ❌ **REJECTED**
- Overkill for our scale
- Pandera is lighter and sufficient
- Adds complexity without proportional value

**Alternative**: Use Pandera + custom assertions in Dataform

**Action**: ❌ **DO NOT ADD**

---

### 11. hmmlearn ❌ **REJECTED**

**What**: Hidden Markov Models for regime detection

**Fit**: ❌ **REJECTED**
- We already have VIX-based regime system
- Manual regimes are domain knowledge (Trump eras)
- Adds complexity without clear benefit

**Action**: ❌ **DO NOT ADD**

---

### 12. Evidently AI ❌ **REJECTED**

**What**: Data drift detection

**Fit**: ❌ **REJECTED**
- We already have segmentation strategy (prevents drift)
- Cloud-dependent (doesn't fit Mac training)
- Overkill for our scale

**Action**: ❌ **DO NOT ADD**

---

### 13. Nixtla API ❌ **REJECTED**

**What**: Time series forecasting API

**Fit**: ❌ **REJECTED**
- We're doing Mac training (not API-based)
- Redundant with our LightGBM/TFT approach
- Adds external dependency

**Action**: ❌ **DO NOT ADD**

---

### 14. Tigramite ⚠️ **RESEARCH ONLY**

**What**: Time-series causal discovery (PCMCI)

**Fit**: ⚠️ **RESEARCH ONLY**
- Validates fundamental assumptions (Big 8 drivers)
- But: Complex, research tool, not production
- Could validate "China → ZL" causality

**Use Case**:
- Research: Validate Big 8 driver causality
- Not production: Too complex for pipeline

**Cost**: Free (open source)

**Decision**: ⚠️ **RESEARCH ONLY** - Use for validation, not production

**Action**: ⚠️ **OPTIONAL** - Add to research scripts only

---

## ✅ Final Approved Stack

### Immediate Additions (High-Value, Low-Bloat):

1. ✅ **Pandas-TA** - Already planned, add to `requirements.txt`
2. ✅ **Pandera** - CRITICAL for logic validation, add immediately
3. ✅ **pycot-reports** - Solves CFTC parsing, add to ingestion script
4. ✅ **wasdeparser** - Solves USDA parsing, add to ingestion script
5. ✅ **SHAP** - Already planned, ensure in `requirements.txt`

**Total Cost**: $0 (all free/open source)

---

### Optional Additions (Research/Validation):

6. ⚠️ **ruptures** - Optional for regime validation
7. ⚠️ **Alphalens-Reloaded** - Optional for factor analysis
8. ⚠️ **Tigramite** - Optional for causal validation

**Decision**: Add to research scripts only, not production pipeline

---

## 📋 Implementation Plan

### Phase 1: Critical Tools (Immediate)

**Add to `requirements.txt`**:
```txt
pandas-ta>=0.3.14b
pandera>=0.18.0
pycot-reports>=0.1.0
wasdeparser>=0.1.0
shap>=0.44.0
```

**Integrate into Scripts**:
1. **Pandera**: Add to `src/training/baselines/lightgbm_zl.py` (input validation)
2. **pycot-reports**: Replace manual parsing in `src/ingestion/cftc/collect_cftc_comprehensive.py`
3. **wasdeparser**: Replace manual parsing in `src/ingestion/usda/collect_usda_comprehensive.py`
4. **SHAP**: Add to `src/training/baselines/lightgbm_zl.py` (post-training analysis)

---

### Phase 2: Research Tools (Optional)

**Add to `requirements-research.txt`** (separate file):
```txt
ruptures>=1.1.8
alphalens-reloaded>=0.5.0
tigramite>=0.6.0
```

**Use Case**: Research/validation scripts only, not production pipeline

---

## ✅ Summary

### Approved (5 tools):
- ✅ Pandas-TA (technical indicators)
- ✅ Pandera (logic validation - CRITICAL)
- ✅ pycot-reports (CFTC parsing)
- ✅ wasdeparser (USDA parsing)
- ✅ SHAP (model interpretability)

### Rejected (5 tools):
- ❌ mlfinlab (commercial, alternatives exist)
- ❌ Great Expectations (overkill, Pandera sufficient)
- ❌ hmmlearn (redundant with VIX regimes)
- ❌ Evidently AI (cloud-dependent, overkill)
- ❌ Nixtla API (redundant with Mac training)

### Optional (3 tools):
- ⚠️ ruptures (research only)
- ⚠️ Alphalens-Reloaded (research only)
- ⚠️ Tigramite (research only)

**Total Cost**: $0 (all approved tools are free)

**Bloat Score**: ✅ **LOW** - Only 5 production tools, all high-value

---

**Last Updated**: November 28, 2025

