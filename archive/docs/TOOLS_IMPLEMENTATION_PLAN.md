# Pre-Built Tools Implementation Plan

**Date**: November 28, 2025  
**Status**: ✅ **APPROVED** - 5 high-value tools, no bloat

---

## ✅ Approved Tools (5)

### 1. Pandas-TA ✅

**Status**: Already planned  
**Action**: Ensure in `requirements.txt`  
**Use**: Mac training (technical indicators validation)

---

### 2. Pandera ✅ **CRITICAL**

**Status**: ✅ **ADD IMMEDIATELY**  
**Action**: 
- Add to `requirements.txt`
- Create `src/training/utils/validation_schema.py`
- Integrate into `src/training/baselines/lightgbm_zl.py`

**Use**: 
- Prevents logic inversions (China sentiment bug)
- Validates training data before model training
- Hard-codes economic assumptions as unit tests

---

### 3. pycot-reports ✅

**Status**: ✅ **ADD IMMEDIATELY**  
**Action**: 
- Add to `requirements.txt`
- Integrate into `src/ingestion/cftc/collect_cftc_comprehensive.py`

**Use**: 
- Replaces manual CFTC COT parsing
- Handles Legacy vs. Disaggregated split automatically
- Saves weeks of work

---

### 4. wasdeparser ✅

**Status**: ✅ **ADD IMMEDIATELY**  
**Action**: 
- Add to `requirements.txt`
- Integrate into `src/ingestion/usda/collect_usda_comprehensive.py`

**Use**: 
- Replaces manual USDA WASDE parsing
- Handles "Revisionist History" (pull specific dates)
- Saves weeks of work

---

### 5. SHAP ✅

**Status**: Already planned  
**Action**: Ensure in `requirements.txt`  
**Use**: 
- Post-training feature importance
- Detects logic inversions (slope analysis)
- Model interpretability

---

## 📋 Implementation Checklist

### Immediate (Before Training):

- [ ] ✅ Add Pandera to `requirements.txt`
- [ ] ✅ Create `src/training/utils/validation_schema.py`
- [ ] ✅ Integrate Pandera validation into `lightgbm_zl.py`
- [ ] ✅ Add pycot-reports to `requirements.txt`
- [ ] ✅ Integrate pycot-reports into CFTC ingestion script
- [ ] ✅ Add wasdeparser to `requirements.txt`
- [ ] ✅ Integrate wasdeparser into USDA ingestion script
- [ ] ✅ Verify SHAP in `requirements.txt`

---

## ✅ Summary

**Approved**: 5 tools (all free, high-value, low-bloat)  
**Rejected**: 5 tools (commercial, overkill, or misaligned)  
**Optional**: 3 tools (research only)

**Total Cost**: $0  
**Bloat Score**: ✅ **LOW**

---

**Last Updated**: November 28, 2025

