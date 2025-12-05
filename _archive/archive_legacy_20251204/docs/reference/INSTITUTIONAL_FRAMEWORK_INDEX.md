---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# Institutional Framework - Documentation Index (V15)

**Date**: November 28, 2025  
**Purpose**: Central index for institutional quant methodology (V15)  
**Status**: V15 architecture; V14 docs marked legacy

---

## Quick Navigation

| Document | Purpose | Priority |
|----------|---------|----------|
| **BEST_PRACTICES.md** | Mandatory rules (no fake data, Mac-only training) | CRITICAL |
| **MASTER_PLAN.md** | System architecture (Mac + BigQuery + Python ETL + Vercel) | CRITICAL |
| **TRAINING_PLAN.md** | Mac M4 training phases & ensembles | CRITICAL |
| **CURSOR_MASTER_INSTRUCTION_SET.md** | Post-move audit protocol (V15-aligned where applicable) | HIGH |
| **archive/py_knowledge/** (LEGACY) | V14/V15 research notes – reference only | LEGACY |

---

## 1. Core Conceptual Frameworks

### Core Conceptual Rules

- **Conviction vs Confidence**  
  - Conceptual distinction remains: conviction = directional clarity; confidence = error bands.  
  - For details, see MASTER_PLAN sentiment/regime sections and any future `CONVICTION_VS_CONFIDENCE.md` (to be written under V15).

- **Signal Treatment Rules**  
  - Principles (yield curve context, VIX regimes, sentiment with mechanism, etc.) still apply.  
  - Until a V15 rewrite exists, treat them as **implicit rules** referenced from MASTER_PLAN/TRAINING_PLAN rather than a standalone doc.

---

## 2. Operational Protocols

### Cursor Master Instruction Set
**File**: `CURSOR_MASTER_INSTRUCTION_SET.md`

**Purpose**: Mandatory audit protocol after major market moves

**Trigger Conditions**:
- VIX > 25
- USD/BRL > 3% move
- ZL > 2% move
- FCPO > 3% move
- Drought > +2σ
- USDA reports
- Policy announcements

**Audit Sequence** (must execute in order):
1. Raw Layer - Integrity checks
2. Curated Layer - Aggregation alignment
3. Training Layer - Feature validation
4. Prediction Layer - Upload + MAPE/Sharpe
5. Dashboard - View resolution

**Canonical Datasets** (ONLY these):
- `raw_intelligence`
- `curated`
- `training`
- `predictions`
- `monitoring`
- `vegas_intelligence`
- `archive`

**Status**: Binding protocol for all AI assistants

---

### Regime Weights Research
**File**: `REGIME_WEIGHTS_RESEARCH.md` (in `scripts/migration/`, legacy but conceptually aligned)

**Research-Based Weights** (50-5000 scale):

| Regime | Weight | Rationale |
|--------|--------|-----------|
| trump_2023_2025 | 5000 | Maximum recency bias |
| structural_events | 2000 | Extreme event learning |
| tradewar_2017_2019 | 1500 | Policy similarity |
| inflation_2021_2023 | 1200 | Current macro |
| covid_2020_2021 | 800 | Supply disruption |
| financial_crisis_2008_2009 | 500 | Volatility learning |
| commodity_crash_2014_2016 | 400 | Crash dynamics |
| qe_supercycle_2010_2014 | 300 | Commodity boom |
| precrisis_2000_2007 | 100 | Baseline patterns |
| historical_pre2000 | 50 | Pattern learning only |

**Principles Applied**:
- Recency bias (100x differential)
- Importance weighting (crisis = high)
- Sample compensation (small but relevant)
- Multiplicative scale (gradient impact)

**Status**: Implemented in `scripts/migration/04_create_regime_tables.sql`

---

## 3. Architecture Alignment

### System Architecture Alignment
**Files**:
- `docs/architecture/MASTER_PLAN.md` – Canonical system architecture.  
- `docs/architecture/TRAINING_PLAN.md` – Canonical training pipeline.  
- `docs/architecture/FINAL_COMPLETE_BQ_SCHEMA.sql` – Canonical BigQuery schema.

Use these three as **primary references** for architecture; older alignment reports (e.g., `ARCHITECTURE_ALIGNMENT_COMPLETE.md`) are considered legacy and may be moved to an archive.

---

## 4. Implementation Status

### Migration Phases

| Phase | Status | Details |
|-------|--------|---------|
| Phase 1: Archive | ✅ Complete | 10 tables → archive.legacy_20251114__* |
| Phase 2: Datasets | ✅ Complete | 7/7 datasets verified |
| Phase 3: Tables | ✅ Complete | 12/12 tables created |
| Phase 4: Scripts | ✅ Complete | 15/15 scripts updated |
| Phase 6: Shim Views | ✅ Complete | 5/5 backward compatibility views |

### Code Updates

| Category | Files | Status |
|----------|-------|--------|
| Export | 1 script | ✅ Complete |
| Training | 13 scripts | ✅ All updated |
| Prediction | 2 scripts | ✅ All updated |
| Upload | 1 script | ✅ Created |

### Critical Fixes

| Issue | Solution | Status |
|-------|----------|--------|
| Regime weights 1000x off | Updated SQL (50-5000) | ✅ Fixed |
| Upload script missing | Created upload_predictions.py | ✅ Created |
| GPT5 outdated | Updated to local-first | ✅ Fixed |
| Model save pattern | Version directories | ✅ Fixed |

---

## 5. Reference Documentation

### Data Sources & Integrations
- **COMPLETE_FEATURE_LIST_290.md** – V14 feature inventory (reference only; V15 features live in MASTER_PLAN).  
- **COMPLETE_SYSTEM_FLOW.md** – To be aligned with V15 schema; treat cautiously until updated.  
- **archive/py_knowledge/** – Detailed PyTorch/quant research notes (read‑only, some V14 context).

### Performance Tracking & Architecture
- Performance metrics, MAPE/Sharpe, and monitoring framework will be consolidated into a dedicated V15 monitoring doc (TBD).  
- Legacy “Training Master Execution Plan” and mapping specs are superseded by MASTER_PLAN + TRAINING_PLAN and may be archived.

---

## 6. Validation & Testing

### Pre-Production Checklist

Before deploying any model or prediction:

**Data Validation**:
- [ ] Raw layer audit passes (Section 2.A)
- [ ] Curated layer audit passes (Section 2.B)
- [ ] Training layer audit passes (Section 2.C)
- [ ] No legacy naming detected
- [ ] Regime weights validated (50-5000)

**Code Validation**:
- [ ] No Vertex AI references in training code
- [ ] No BQML training code
- [ ] All scripts use new naming convention
- [ ] Model metadata files present
- [ ] Predictions uploaded successfully

**Conceptual Validation**:
- [ ] Conviction and confidence separated
- [ ] Signals paired with validators
- [ ] Regime-specific MAPE known
- [ ] Error bands widen in crisis
- [ ] No false precision claims

---

## 7. Monitoring & Alerts

### Daily Checks (Automated)

**Data Quality**:
```bash
# Run every morning at 6am
python scripts/audits/daily_data_quality_check.py
```

**Checks**:
- Timestamp gaps in raw layer
- Aggregate staleness
- Prediction upload status
- MAPE/Sharpe recalculation

### Post-Move Checks (Event-Driven)

**Trigger Detection**:
```bash
# Run every 15 minutes during market hours
python scripts/monitoring/market_move_detector.py
```

**If triggered**:
```bash
# Execute full audit sequence
python scripts/audits/post_move_audit.py --full
```

---

## 8. Emergency Procedures

### Data Corruption Detected

1. **Quarantine immediately**: Move bad rows to `raw_intelligence.quarantine_scrapes`
2. **Alert**: Page on-call, log to `monitoring.data_quality_events`
3. **Halt downstream**: Stop prediction generation
4. **Investigate**: Trace to ingestion source
5. **Backfill**: From alternative source
6. **Validate**: Rerun full audit
7. **Resume**: Only after validation passes

### Prediction Staleness

1. **Check local models**: Ensure prediction files exist
2. **Regenerate if needed**: `python src/prediction/generate_local_predictions.py --horizon all`
3. **Upload**: `python scripts/upload_predictions.py`
4. **Verify**: Check `predictions.vw_zl_{h}_latest`
5. **Recalculate metrics**: MAPE + Sharpe

### Training Data Mismatch

1. **Reexport from BigQuery**: `python scripts/export_training_data.py --surface prod --horizon all`
2. **Validate exports**: Check file sizes, row counts
3. **Retrain critical models**: Priority = current horizon
4. **Regenerate predictions**: After retraining
5. **Upload + validate**: Full prediction pipeline

---

## Quick Reference Card

### Post-Move Audit (One Command)

```bash
# Execute full audit after major market move
python scripts/audits/post_move_audit.py \
  --trigger "VIX_SPIKE" \
  --severity "CRITICAL" \
  --auto-remediate
```

### Manual Validation

```bash
# Check raw layer
python scripts/audits/validate_raw_layer.py --days 7

# Check training tables
python scripts/audits/validate_training_tables.py

# Check predictions
python scripts/audits/validate_predictions.py --horizons all

# Full pipeline test
python scripts/audits/end_to_end_validation.py
```

---

## Integration with Existing Documentation

This framework integrates with:

**Architecture**:
- `ARCHITECTURE_ALIGNMENT_COMPLETE.md` - System alignment
- `GPT5_READ_FIRST.md` - Current vs legacy
- `TRAINING_MASTER_EXECUTION_PLAN.md` - Training strategy

**Migration**:
- `scripts/migration/PHASE_1_3_COMPLETION_REPORT.md` - Migration status
- `scripts/migration/MIGRATION_STATUS.md` - Phase tracking
- `docs/migrations/20251114_NAMING_ARCHITECTURE_MIGRATION.md` - Execution log

**Data Quality**:
- `docs/handoffs/CRITICAL_CONTEXT_READ_FIRST.md` - TE corruption
- `docs/reference/V14_METADATA_EXPANSION.md` - Schema standards
- `docs/audits/COMPLETE_DATASET_INVENTORY_20251114.md` - Full inventory

---

## Conclusion

**The Framework**: 
1. Separate conviction from confidence
2. Treat signals as mechanisms, not numbers
3. Audit after every major move
4. Consolidate to canonical datasets only

**The Discipline**:
- No shortcuts
- No new datasets
- No skipped validations
- No false precision

**The Edge**: Institutional-grade rigor that amateur systems lack.

---

**Last Updated**: November 14, 2025  
**Documentation Status**: COMPLETE  
**Implementation Status**: Framework defined, automation pending  
**Next**: Build automated audit execution + monitoring scripts
