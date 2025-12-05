# Model Retraining Procedures

**Status:** Production  
**Last Updated:** December 3, 2025

## Retraining Schedule

**Weekly:** Ensemble weight recalculation (every Monday)

**Monthly:** Full model retrain on expanding window (first Sunday of month)

**Quarterly:** Model selection review and pruning (manual)

## Weekly Ensemble Weight Update

### Procedure
1. Query last 30 days of forecast vs. actuals
2. Compute MAE per model per horizon
3. Recalculate inverse MAPE weights
4. Update `model_registry.ensemble_weight`
5. Test new weights on validation set
6. Deploy if improvement ≥ 0.5% MAPE

### Location
- Script: `Scripts/training/update_ensemble_weights.py`
- Scheduled: Vercel Cron (Monday 2 AM)

## Monthly Full Retrain

### Procedure
1. Export latest training data (all available history)
2. Run all 31 models × 5 horizons = 155 training runs
3. Evaluate on holdout set (last 60 days)
4. Update `training_runs` table with new metrics
5. Select new top performers if beating current by ≥ 1% MAPE
6. Archive old models to `models/archive/YYYYMM/`

### Duration
- Expected: 4-6 hours (sequential training)
- Resource: Local Mac M4 or MotherDuck compute

### Location
- Script: `Scripts/training/monthly_retrain.py`
- Triggered: Manual (first Sunday of month)

## Quarterly Model Review

### Checklist
- [ ] Review feature importance drift
- [ ] Evaluate if any of 31 models consistently underperform
- [ ] Consider adding new models (if AnoFox updates)
- [ ] Review bucket definitions (Chris feedback)
- [ ] Update documentation

### Location
- Documentation: This file
- Review meetings: First week of quarter

