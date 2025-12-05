# Regime Classification System Documentation

**Date:** December 3, 2024  
**Source:** `reference.regime_calendar`, `reference.regime_weights`  
**Status:** Documentation in progress

---

## Overview

The regime classification system assigns market periods to specific regimes (buckets) and applies regime-specific weights during training. This system is critical for preserving the intelligence from CBI-V15.

---

## Regime Types

Based on `build_daily_ml_matrix.py`, the system uses range-based regime classification:

**Regime Types (from code):**
- Regimes are defined by date ranges in `reference.regime_calendar`
- Each regime has a `regime_type`, `start_date`, `end_date`, and `base_weight`
- Regimes are assigned to each date in the training data

---

## Regime Weights

**Weight Scale:** 50-1000 (as per MASTER_PLAN.md)

**Weight Application:**
- Base weights come from `reference.regime_weights` table
- Weights are applied during training to emphasize certain periods
- Higher weights = more emphasis in training

**Example Regimes (from MASTER_PLAN.md):**
- Trump 2.0 (2023–2025): weight ×5000
- Trade War (2017–2019): weight ×1500
- Inflation (2021–2022): weight ×1200
- Crisis (2008, 2020): weight ×500–800
- Historical (<2000): weight ×50

---

## Bucket Definitions

**7 News Buckets (from integration plan):**
1. Biofuel Policy
2. China Demand
3. Trade Relations
4. Weather/Supply
5. Macro/Financial
6. Geopolitical
7. Logistics/Infrastructure

**Note:** These buckets may be separate from regime classification - need to verify in actual data.

---

## Implementation

### Regime Assignment
Regimes are assigned to each row in `daily_ml_matrix` via:
```python
regimes = load_table("""
    SELECT regime_type as regime, start_date, end_date, base_weight
    FROM reference.regime_calendar
""")

# Join regimes to daily data based on date ranges
df = df.merge(regimes, on='date', how='left')
```

### Weight Application
Weights are applied during training:
- `regime_weight` column in daily_ml_matrix
- Used to weight training samples
- Capped at 1000 (per MASTER_PLAN.md)

---

## Migration to DuckDB

**Status:** Tables loaded successfully

**Tables:**
- `reference.regime_calendar` - Regime date ranges
- `reference.regime_weights` - Regime-specific weights

**Verification:**
- [ ] Verify regime_calendar has data
- [ ] Verify regime_weights has data
- [ ] Test regime assignment logic with DuckDB data

---

## Dependencies

**Input Tables:**
- `reference.regime_calendar` - Regime definitions
- `reference.regime_weights` - Weight multipliers

**Output:**
- `regime` column in `daily_ml_matrix`
- `regime_weight` column in `daily_ml_matrix`

---

**Last Updated:** December 3, 2024

