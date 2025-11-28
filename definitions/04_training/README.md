# Training Layer

This directory contains training-ready tables with targets, horizons, and regime weights.

## Purpose

Training tables join features with targets (future price levels) and regime weights for Mac M4 training.

## Tables

- `zl_training_1w.sqlx` - 1 week horizon (5 trading days)
- `zl_training_1m.sqlx` - 1 month horizon (20 trading days)
- `zl_training_3m.sqlx` - 3 month horizon (60 trading days)
- `zl_training_6m.sqlx` - 6 month horizon (120 trading days)

## Structure

Each training table includes:
- **Features**: All features from `daily_ml_matrix`
- **Targets**: Future price levels (P_{t+N})
- **Regime Weights**: Evidence-based weights for training
- **Date Range**: Point-in-time discipline (no lookahead)

## Targets

Targets are future price levels, not returns:
- `target_zl_1w` = Price at t+5 trading days
- `target_zl_1m` = Price at t+20 trading days
- `target_zl_3m` = Price at t+60 trading days
- `target_zl_6m` = Price at t+120 trading days

## Regime Weights

Weights based on VIX levels and shock multipliers:
- Base weight = avg(VIX in regime) * 100
- Capped at 3000 to avoid overfit
- Shock multipliers: 0.15 for policy, vol, supply, geopol

## Train/Val/Test Splits

- **Train**: 2000-01-01 to 2023-01-01
- **Validation**: 2023-01-01 to 2024-01-01
- **Test**: 2024-01-01 to present

## Export for Mac Training

Training tables exported to Parquet for Mac M4 training:

```bash
python scripts/export/export_training_data.py --horizon 1m
```

Output: `TrainingData/exports/zl_training_1m.parquet`

---

**Last Updated**: November 28, 2025

